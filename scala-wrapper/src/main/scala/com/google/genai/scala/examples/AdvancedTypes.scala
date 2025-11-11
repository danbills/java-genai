package com.google.genai.scala.examples

import cats.effect.*
import com.google.genai.scala.client.*
import com.google.genai.scala.types.*
import com.google.genai.scala.types.PhantomTypes.*
import com.google.genai.scala.constraints.StringConstraints.*
import com.google.genai.scala.constraints.NumericConstraints.*

/**
 * Advanced examples demonstrating the full power of the type system.
 *
 * This example shows:
 * 1. Type-level model capability constraints
 * 2. Phantom type API variant separation
 * 3. ADT exhaustive pattern matching
 * 4. Iron type validation chains
 * 5. Zero-cost opaque type abstractions
 */
object AdvancedTypes extends IOApp.Simple:

  def run: IO[Unit] =
    for
      _ <- demonstrateCapabilityConstraints
      _ <- demonstrateApiVariantSeparation
      _ <- demonstrateAdtPatternMatching
      _ <- demonstrateIronValidationChains
      _ <- demonstrateOpaqueTypes
    yield ()

  // ============================================================================
  // 1. TYPE-LEVEL MODEL CAPABILITY CONSTRAINTS
  // ============================================================================

  /**
   * The type system prevents using models for unsupported operations.
   * This is caught at COMPILE TIME, not runtime!
   */
  def demonstrateCapabilityConstraints: IO[Unit] =
    IO.println("=== Capability Constraints ===") *> IO.pure {
      // Text generation models
      type TextModel = Model[GeminiApi, TextGeneration & FunctionCalling & CodeExecution]
      val gemini20: TextModel = Model.gemini20Flash
      val gemini15: TextModel = Model.gemini15Pro

      // Image generation model
      type ImageModel = Model[GeminiApi, ImageGeneration]
      val imagen: ImageModel = Model.imagen3

      // Embedding model
      type EmbedModel = Model[GeminiApi, Embeddings]
      val embedModel: EmbedModel = Model.textEmbedding

      // These type signatures ENSURE correct usage:
      def textOnly[A <: ApiVariant, C <: ModelCapability](
        model: Model[A, C]
      )(using HasCapability[C, TextGeneration]): String =
        s"Can generate text with ${model.id.value}"

      def imageOnly[A <: ApiVariant, C <: ModelCapability](
        model: Model[A, C]
      )(using HasCapability[C, ImageGeneration]): String =
        s"Can generate images with ${model.id.value}"

      def embedOnly[A <: ApiVariant, C <: ModelCapability](
        model: Model[A, C]
      )(using HasCapability[C, Embeddings]): String =
        s"Can generate embeddings with ${model.id.value}"

      // ✅ These compile: correct capabilities
      println(textOnly(gemini20))
      println(imageOnly(imagen))
      println(embedOnly(embedModel))

      // ❌ These DON'T compile: capability mismatch
      // println(textOnly(imagen))      // imagen doesn't have TextGeneration
      // println(imageOnly(gemini20))   // gemini20 doesn't have ImageGeneration
      // println(embedOnly(gemini20))   // gemini20 doesn't have Embeddings
    }

  // ============================================================================
  // 2. PHANTOM TYPE API VARIANT SEPARATION
  // ============================================================================

  /**
   * Phantom types prevent mixing Gemini and Vertex AI operations.
   * The type parameter tracks which API variant you're using.
   */
  def demonstrateApiVariantSeparation: IO[Unit] =
    IO.println("\n=== API Variant Separation ===") *> IO.pure {

      // Function that works with ANY API variant
      def universalOperation[A <: ApiVariant](
        client: GenAiClient[A],
        model: Model[A, TextGeneration]
      ): IO[String] =
        IO.pure(s"This works with any API variant")

      // Function that ONLY works with Gemini API
      def geminiOnlyOperation(client: GeminiApiClient): IO[FileId] =
        client.uploadFile(
          content = "test".getBytes,
          mimeType = MimeType.TextPlain
        )

      // Function that ONLY works with Vertex AI
      def vertexOnlyOperation(client: VertexApiClient): IO[TuningJobId] =
        client.createTuningJob(
          baseModel = ModelId.Gemini_1_5_Flash,
          trainingDataUri = Uri.unsafe("gs://bucket/data.jsonl"),
          config = TuningConfig(epochs = 10)
        )

      // Type system ensures you can't mix them:
      val geminiClient: IO[GeminiApiClient] = ???
      val vertexClient: IO[VertexApiClient] = ???

      // ✅ These compile: correct API variant
      // geminiClient.flatMap(geminiOnlyOperation)
      // vertexClient.flatMap(vertexOnlyOperation)

      // ❌ These DON'T compile: API variant mismatch
      // geminiClient.flatMap(vertexOnlyOperation)  // Can't use Vertex operation on Gemini client
      // vertexClient.flatMap(geminiOnlyOperation)  // Can't use Gemini operation on Vertex client

      println("API variants are enforced at compile-time!")
    }

  // ============================================================================
  // 3. ADT EXHAUSTIVE PATTERN MATCHING
  // ============================================================================

  /**
   * ADTs (Algebraic Data Types) with sealed traits ensure exhaustive matching.
   * The compiler FORCES you to handle all cases.
   */
  def demonstrateAdtPatternMatching: IO[Unit] =
    IO.println("\n=== ADT Exhaustive Pattern Matching ===") *> IO {

      // Example: Processing different part types
      def processPart(part: Part): String = part match
        case Part.Text(text) =>
          s"📝 Text: ${text.value.take(50)}..."

        case Part.InlineData(data, mimeType) =>
          s"📎 Inline ${mimeType.value}: ${data.value.take(20)}..."

        case Part.FileData(uri, mimeType) =>
          s"📁 File ${mimeType.value}: ${uri.value}"

        case Part.FunctionCall(name, args) =>
          s"⚙️  Function call: ${name.value}(${args.size} args)"

        case Part.FunctionResponse(name, response) =>
          s"✅ Function response: ${name.value}"

        case Part.ExecutableCode(language, code) =>
          s"💻 ${language} code: ${code.value.take(30)}..."

        case Part.CodeExecutionResult(outcome, output) =>
          val emoji = outcome match
            case ExecutionOutcome.Success => "✅"
            case ExecutionOutcome.Failure => "❌"
          s"$emoji Execution result: ${output.take(30)}..."

        case Part.Thought(text) =>
          s"🧠 Model thinking: ${text.value.take(50)}..."

        // Compiler ensures ALL cases are handled!
        // If we add a new Part type, this will fail to compile until we handle it

      // Example: Processing finish reasons
      def describeFinish(reason: FinishReason): String = reason match
        case FinishReason.Stop => "✅ Completed naturally"
        case FinishReason.MaxTokens => "⚠️  Hit token limit"
        case FinishReason.Safety => "🛑 Blocked by safety filter"
        case FinishReason.Recitation => "🛑 Blocked by recitation check"
        case FinishReason.Blocklist => "🛑 Blocked by blocklist"
        case FinishReason.ProhibitedContent => "🛑 Prohibited content"
        case FinishReason.Spii => "🛑 Sensitive personal information"
        case FinishReason.Other => "❓ Other reason"

      // Example: Processing tuning job status
      def describeTuningStatus(status: TuningJobStatus): String = status match
        case TuningJobStatus.Pending => "⏳ Waiting to start"
        case TuningJobStatus.Running => "🔄 Training in progress"
        case TuningJobStatus.Succeeded => "✅ Training completed"
        case TuningJobStatus.Failed => "❌ Training failed"
        case TuningJobStatus.Cancelled => "🛑 Training cancelled"

      val testParts = List(
        Part.Text(Prompt.unsafe("Hello, world!")),
        Part.FunctionCall(NonEmptyString.unsafe("search"), Map("query" -> "cats")),
        Part.Thought(NonEmptyString.unsafe("Let me think about this..."))
      )

      testParts.foreach(part => println(processPart(part)))
      println(describeFinish(FinishReason.Stop))
      println(describeTuningStatus(TuningJobStatus.Running))
    }

  // ============================================================================
  // 4. IRON VALIDATION CHAINS
  // ============================================================================

  /**
   * Iron types can be chained with Either/for-comprehensions
   * for complex validation workflows.
   */
  def demonstrateIronValidationChains: IO[Unit] =
    IO.println("\n=== Iron Validation Chains ===") *> IO {

      // Validate entire configuration in one chain
      def createValidatedConfig(
        tempValue: Double,
        topPValue: Double,
        topKValue: Int,
        maxTokensValue: Int,
        candidatesValue: Int
      ): Either[String, GenerationConfig] =
        for
          temp <- Temperature(tempValue)
          topP <- TopP(topPValue)
          topK <- TopK(topKValue)
          maxTokens <- MaxOutputTokens(maxTokensValue)
          candidates <- CandidateCount(candidatesValue)
        yield GenerationConfig(
          temperature = Some(temp),
          topP = Some(topP),
          topK = Some(topK),
          maxOutputTokens = Some(maxTokens),
          candidateCount = Some(candidates)
        )

      // ✅ Valid configuration
      createValidatedConfig(1.0, 0.9, 40, 1000, 3) match
        case Right(config) => println(s"✅ Valid config created")
        case Left(err) => println(s"❌ $err")

      // ❌ Invalid temperature
      createValidatedConfig(5.0, 0.9, 40, 1000, 3) match
        case Right(_) => println("Unexpected success")
        case Left(err) => println(s"❌ Temperature validation failed: $err")

      // ❌ Invalid topP
      createValidatedConfig(1.0, 1.5, 40, 1000, 3) match
        case Right(_) => println("Unexpected success")
        case Left(err) => println(s"❌ TopP validation failed: $err")

      // ❌ Invalid candidate count
      createValidatedConfig(1.0, 0.9, 40, 1000, 10) match
        case Right(_) => println("Unexpected success")
        case Left(err) => println(s"❌ Candidate count validation failed: $err")

      // Chain multiple validations together
      val fullValidation: Either[String, (ApiKey, ProjectId, Location, GenerationConfig)] =
        for
          apiKey <- ApiKey("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
          projectId <- ProjectId("my-project-123")
          location <- Location("us-central1")
          config <- createValidatedConfig(0.8, 0.9, 40, 1000, 2)
        yield (apiKey, projectId, location, config)

      fullValidation match
        case Right((key, proj, loc, cfg)) =>
          println(s"✅ Full validation passed")
          println(s"   API Key: ${key.masked}")
          println(s"   Project: ${proj.value}")
          println(s"   Location: ${loc.value}")
        case Left(err) =>
          println(s"❌ Validation failed: $err")
    }

  // ============================================================================
  // 5. ZERO-COST OPAQUE TYPE ABSTRACTIONS
  // ============================================================================

  /**
   * Opaque types provide type safety WITHOUT runtime overhead.
   * They're erased at runtime but enforced at compile-time.
   */
  def demonstrateOpaqueTypes: IO[Unit] =
    IO.println("\n=== Zero-Cost Opaque Types ===") *> IO {

      // At runtime, these are just primitives:
      val temp: Temperature = Temperature.unsafe(1.0) // Runtime: just a Double
      val topP: TopP = TopP.unsafe(0.9)               // Runtime: just a Double
      val topK: TopK = TopK.unsafe(40)                // Runtime: just an Int

      // But at compile-time, they're distinct types:
      def needsTemperature(t: Temperature): Double = t.value
      def needsTopP(p: TopP): Double = p.value
      def needsTopK(k: TopK): Int = k.value

      // ✅ These compile: correct types
      println(s"Temperature: ${needsTemperature(temp)}")
      println(s"TopP: ${needsTopP(topP)}")
      println(s"TopK: ${needsTopK(topK)}")

      // ❌ These DON'T compile: type mismatch
      // println(needsTemperature(topP))  // Can't pass TopP where Temperature expected
      // println(needsTopP(topK))         // Can't pass TopK where TopP expected

      // Opaque types also prevent accidental mixing:
      val apiKey: ApiKey = ApiKey.unsafe("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ")
      val projectId: ProjectId = ProjectId.unsafe("my-project")

      def needsApiKey(key: ApiKey): String = key.masked
      def needsProjectId(id: ProjectId): String = id.value

      // ❌ Can't mix string-based opaque types
      // needsApiKey(projectId)  // Won't compile!
      // needsProjectId(apiKey)  // Won't compile!

      println(s"API Key: ${needsApiKey(apiKey)}")
      println(s"Project: ${needsProjectId(projectId)}")

      // Additional benefits: semantic operations
      val fileSize: FileSizeBytes = FileSizeBytes.unsafe(1_073_741_824L)
      println(s"File size: ${fileSize.value} bytes")
      println(s"File size: ${fileSize.toMB} MB")
      println(s"File size: ${fileSize.toGB} GB")

      val embedding1 = Embedding(Vector(0.1, 0.2, 0.3, 0.4))
      val embedding2 = Embedding(Vector(0.2, 0.3, 0.4, 0.5))
      val normalized = embedding1.normalize
      val similarity = embedding1.cosineSimilarity(embedding2)
      println(f"Embedding similarity: $similarity%.4f")

      println("\n✅ All type operations are ZERO-COST at runtime!")
      println("   Opaque types are erased to primitives")
      println("   But provide full type safety at compile-time")
    }

/**
 * Type-level proofs and evidence.
 *
 * These demonstrate how the compiler uses type-level programming
 * to enforce constraints without any runtime overhead.
 */
object TypeLevelProofs:

  /**
   * Example: prove at compile-time that a model has specific capabilities.
   */
  def proveCapabilities[A <: ApiVariant]: Unit =
    import Model.*

    // Gemini 2.0 Flash has multiple capabilities
    type Gemini20Caps = TextGeneration & FunctionCalling & CodeExecution
    val model: Model[A, Gemini20Caps] = gemini20Flash

    // We can prove it has TextGeneration
    summon[HasCapability[Gemini20Caps, TextGeneration]]

    // We can prove it has FunctionCalling
    summon[HasCapability[Gemini20Caps, FunctionCalling]]

    // We can prove it has CodeExecution
    summon[HasCapability[Gemini20Caps, CodeExecution]]

    // But we CANNOT prove it has ImageGeneration (won't compile)
    // summon[HasCapability[Gemini20Caps, ImageGeneration]]  // Compile error!

  /**
   * Example: prove at compile-time that features are available in API variants.
   */
  def proveFeatureAvailability: Unit =
    // Files feature is available in Gemini API
    summon[FeatureAvailable[Feature.Files.type, GeminiApi]]

    // But NOT in Vertex AI (won't compile)
    // summon[FeatureAvailable[Feature.Files.type, VertexApi]]  // Compile error!

    // Tuning is available in Vertex AI
    summon[FeatureAvailable[Feature.Tuning.type, VertexApi]]

    // But NOT in Gemini API (won't compile)
    // summon[FeatureAvailable[Feature.Tuning.type, GeminiApi]]  // Compile error!

    // Batches available in both
    summon[FeatureAvailable[Feature.Batches.type, GeminiApi]]
    summon[FeatureAvailable[Feature.Batches.type, VertexApi]]
