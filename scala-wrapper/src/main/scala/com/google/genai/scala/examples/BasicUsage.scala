package com.google.genai.scala.examples

import cats.effect.*
import com.google.genai.scala.client.*
import com.google.genai.scala.types.*
import com.google.genai.scala.types.Model.*
import com.google.genai.scala.constraints.StringConstraints.*
import com.google.genai.scala.constraints.NumericConstraints.*

/**
 * Examples demonstrating ultra-constrained API usage with iron types.
 *
 * Key benefits:
 * 1. Compile-time validation of literals
 * 2. Runtime validation with clear error messages
 * 3. Impossible to mix Gemini/Vertex API operations
 * 4. Type-safe model capabilities
 * 5. No primitive obsession
 */
object BasicUsage extends IOApp.Simple:

  def run: IO[Unit] = geminiExamples *> vertexExamples *> constraintExamples

  // ============================================================================
  // GEMINI API EXAMPLES
  // ============================================================================

  def geminiExamples: IO[Unit] =
    GenAiClient.geminiFromEnv().use { client =>
      for
        _ <- IO.println("=== Gemini API Examples ===")
        _ <- basicTextGeneration(client)
        _ <- configuredGeneration(client)
        _ <- imageGeneration(client)
        // _ <- fileUpload(client) // Only compiles with GeminiApiClient!
      yield ()
    }

  def basicTextGeneration(client: GeminiApiClient): IO[Unit] =
    for
      // Prompt is iron-constrained: must be non-empty
      prompt <- IO.fromEither(
        Prompt("Explain quantum computing in one sentence")
          .left.map(err => new RuntimeException(err))
      )

      // Model type ensures text generation capability
      response <- client.generateContent(
        model = gemini20Flash,
        prompt = prompt
      )

      _ <- IO.println(s"Response: ${response.firstText.getOrElse("No text")}")
      _ <- response.usageMetadata.traverse { usage =>
        IO.println(s"Tokens used: ${usage.totalTokenCount}")
      }
    yield ()

  def configuredGeneration(client: GeminiApiClient): IO[Unit] =
    for
      prompt <- IO.pure(Prompt.unsafe("Write a haiku about programming"))

      // Iron types enforce valid ranges at compile-time for literals
      config = GenerationConfig(
        temperature = Some(Temperature.unsafe(1.2)), // 0.0-2.0
        topP = Some(TopP.unsafe(0.9)),                 // 0.0-1.0
        topK = Some(TopK.unsafe(40)),                  // > 0
        maxOutputTokens = Some(MaxOutputTokens.unsafe(100)), // > 0
        candidateCount = Some(CandidateCount.unsafe(3))      // 1-8
      )

      // Or validate at runtime
      validatedConfig <- IO.fromEither(
        for
          temp <- Temperature(0.8)
          topP <- TopP(0.85)
          topK <- TopK(20)
        yield GenerationConfig(
          temperature = Some(temp),
          topP = Some(topP),
          topK = Some(topK)
        )
      ).adaptError { case err: String => new RuntimeException(err) }

      response <- client.generateContent(
        model = gemini15Pro,
        prompt = prompt,
        config = validatedConfig,
        safetySettings = SafetySetting.strictAll
      )

      _ <- IO.println(s"Haiku:\n${response.firstText.getOrElse("")}")
    yield ()

  def imageGeneration(client: GeminiApiClient): IO[Unit] =
    for
      prompt <- IO.pure(Prompt.unsafe("A serene mountain landscape at sunset"))

      // Image generation config with iron-constrained aspect ratio
      config = ImageGenerationConfig(
        numberOfImages = CandidateCount.unsafe(2),
        aspectRatio = Some(AspectRatio.Landscape_16_9),
        negativePrompt = Some(Prompt.unsafe("blurry, low quality")),
        personGeneration = PersonGeneration.DontAllow
      )

      // This only compiles because imagen3 has ImageGeneration capability
      images <- client.generateImages(
        model = imagen3,
        prompt = prompt,
        config = config
      )

      _ <- IO.println(s"Generated ${images.length} images")
      _ <- images.traverse_ { img =>
        IO.println(s"  - ${img.mimeType.value}, ${img.data.value.take(50)}...")
      }
    yield ()

  // This is only available on GeminiApiClient!
  def fileUpload(client: GeminiApiClient): IO[Unit] =
    for
      fileContent = "Hello, world!".getBytes

      fileId <- client.uploadFile(
        content = fileContent,
        mimeType = MimeType.TextPlain,
        displayName = Some(NonEmptyString.unsafe("test.txt"))
      )

      _ <- IO.println(s"Uploaded file: ${fileId.value}")

      metadata <- client.getFile(fileId)
      _ <- IO.println(s"File size: ${metadata.sizeBytes.toMB} MB")

      _ <- client.deleteFile(fileId)
      _ <- IO.println("File deleted")
    yield ()

  // ============================================================================
  // VERTEX AI EXAMPLES
  // ============================================================================

  def vertexExamples: IO[Unit] =
    // Vertex AI requires project ID and location (both iron-constrained)
    (for
      projectId <- ProjectId("my-project-123")
      location = Location.USCentral1
      token <- NonEmptyString("mock-access-token")
    yield (projectId, location, token)) match
      case Right((projectId, location, token)) =>
        GenAiClient.vertex(projectId, location, token).use { client =>
          for
            _ <- IO.println("\n=== Vertex AI Examples ===")
            _ <- vertexTextGeneration(client)
            // _ <- tuningJob(client) // Only compiles with VertexApiClient!
          yield ()
        }
      case Left(err) =>
        IO.println(s"Invalid credentials: $err")

  def vertexTextGeneration(client: VertexApiClient): IO[Unit] =
    for
      prompt <- IO.pure(Prompt.unsafe("What is the capital of France?"))

      response <- client.generateContent(
        model = gemini20Flash,
        prompt = prompt,
        config = GenerationConfig.deterministic
      )

      _ <- IO.println(s"Response: ${response.firstText.getOrElse("No text")}")
    yield ()

  // This is only available on VertexApiClient!
  def tuningJob(client: VertexApiClient): IO[Unit] =
    for
      baseModel = ModelId.Gemini_1_5_Flash
      trainingData <- IO.pure(Uri.unsafe("gs://my-bucket/training-data.jsonl"))

      config = TuningConfig(
        epochs = 10,
        learningRate = Some(0.001),
        batchSize = Some(32)
      )

      jobId <- client.createTuningJob(baseModel, trainingData, config)
      _ <- IO.println(s"Tuning job created: ${jobId.value}")

      status <- client.getTuningJob(jobId)
      _ <- IO.println(s"Status: $status")
    yield ()

  // ============================================================================
  // CONSTRAINT VALIDATION EXAMPLES
  // ============================================================================

  def constraintExamples: IO[Unit] =
    for
      _ <- IO.println("\n=== Constraint Validation Examples ===")

      // Valid temperature
      _ <- Temperature(1.0) match
        case Right(temp) => IO.println(s"✓ Valid temperature: ${temp.value}")
        case Left(err) => IO.println(s"✗ $err")

      // Invalid temperature (too high)
      _ <- Temperature(3.0) match
        case Right(temp) => IO.println(s"✓ Valid temperature: ${temp.value}")
        case Left(err) => IO.println(s"✗ Invalid temperature: $err")

      // Invalid temperature (negative)
      _ <- Temperature(-0.5) match
        case Right(temp) => IO.println(s"✓ Valid temperature: ${temp.value}")
        case Left(err) => IO.println(s"✗ Invalid temperature: $err")

      // Valid API key
      _ <- ApiKey("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890") match
        case Right(key) => IO.println(s"✓ Valid API key: ${key.masked}")
        case Left(err) => IO.println(s"✗ $err")

      // Invalid API key (too short)
      _ <- ApiKey("short") match
        case Right(key) => IO.println(s"✓ Valid API key: ${key.masked}")
        case Left(err) => IO.println(s"✗ Invalid API key: $err")

      // Valid project ID
      _ <- ProjectId("my-project-123") match
        case Right(id) => IO.println(s"✓ Valid project ID: ${id.value}")
        case Left(err) => IO.println(s"✗ $err")

      // Invalid project ID (uppercase not allowed)
      _ <- ProjectId("My-Project-123") match
        case Right(id) => IO.println(s"✓ Valid project ID: ${id.value}")
        case Left(err) => IO.println(s"✗ Invalid project ID: $err")

      // Valid MIME type
      _ <- MimeType("image/png") match
        case Right(mime) =>
          IO.println(s"✓ Valid MIME type: ${mime.value}, is image: ${mime.isImage}")
        case Left(err) => IO.println(s"✗ $err")

      // Invalid MIME type
      _ <- MimeType("not-a-mime-type") match
        case Right(mime) => IO.println(s"✓ Valid MIME type: ${mime.value}")
        case Left(err) => IO.println(s"✗ Invalid MIME type: $err")

      // Aspect ratio calculation
      _ <- AspectRatio(1920, 1080) match
        case Right(ratio) => IO.println(s"✓ Aspect ratio 1920x1080 = ${ratio.value}")
        case Left(err) => IO.println(s"✗ $err")

      // File size conversion
      fileSize = FileSizeBytes.unsafe(1_073_741_824L) // 1 GB
      _ <- IO.println(s"File size: ${fileSize.value} bytes = ${fileSize.toGB} GB")

    yield ()

  // ============================================================================
  // COMPILE-TIME SAFETY EXAMPLES
  // ============================================================================

  // These won't compile - demonstrating type safety:

  /*
  def typeErrors(geminiClient: GeminiApiClient, vertexClient: VertexApiClient): IO[Unit] =
    for
      // ERROR: Can't call vertex-only operations on Gemini client
      // _ <- geminiClient.createTuningJob(...)

      // ERROR: Can't call gemini-only operations on Vertex client
      // _ <- vertexClient.uploadFile(...)

      // ERROR: Can't use text generation model for image generation
      // _ <- geminiClient.generateImages(
      //   model = gemini20Flash, // This has TextGeneration capability, not ImageGeneration
      //   prompt = Prompt.unsafe("test")
      // )

      // ERROR: Can't use image generation model for text
      // _ <- geminiClient.generateContent(
      //   model = imagen3, // This has ImageGeneration capability, not TextGeneration
      //   prompt = Prompt.unsafe("test")
      // )

      // ERROR: Can't create Temperature outside valid range (0.0-2.0)
      // config = GenerationConfig(temperature = Some(Temperature.unsafe(5.0)))
      // This compiles but will throw at runtime. Better to use Temperature.apply for validation:
      // temp <- Temperature(5.0) // Returns Either[String, Temperature]

      // ERROR: Can't create CandidateCount outside valid range (1-8)
      // count = CandidateCount.unsafe(10)

      // ERROR: Can't pass empty string where NonEmptyString is required
      // prompt = Prompt.unsafe("") // Will throw IllegalArgumentException at runtime
      // Better: Prompt("") returns Left("Should be a non empty string")

    yield ()
  */
