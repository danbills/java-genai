# GenAI Scala Ultra-Constrained Wrapper

An **ultra-constrained** Scala 3 wrapper over Google's GenAI Java SDK, using [Iron](https://github.com/Iltotore/iron) refinement types and phantom types for maximum compile-time safety.

## 🔒 Ultra-Constraint Philosophy

This wrapper eliminates entire classes of bugs through:

1. **Iron Refinement Types**: Compile-time and runtime validation of all numeric/string constraints
2. **Phantom Types**: API variant (Gemini/Vertex) enforced at compile-time
3. **Capability Types**: Model capabilities (text/image/video generation) enforced at type-level
4. **ADTs**: Exhaustive pattern matching for all domain types
5. **Opaque Types**: Zero-cost abstractions that prevent primitive obsession
6. **No Nulls**: Optional values everywhere using `Option`

### What Can't Compile

```scala
// ❌ Can't call Vertex AI operations on Gemini client
geminiClient.createTuningJob(...) // COMPILE ERROR

// ❌ Can't call Gemini-only operations on Vertex client
vertexClient.uploadFile(...) // COMPILE ERROR

// ❌ Can't use wrong model for operation
client.generateImages(
  model = gemini20Flash, // Has TextGeneration, not ImageGeneration
  prompt = "..."
) // COMPILE ERROR

// ❌ Can't use invalid temperature
Temperature.unsafe(5.0) // Runtime error (use Temperature.apply for Either)

// ❌ Can't create invalid project ID
ProjectId("INVALID_FORMAT") // Either[String, ProjectId] = Left(...)

// ❌ Can't mix API types
val geminiModel: Model[GeminiApi, TextGeneration] = ...
val vertexClient: VertexApiClient = ...
vertexClient.generateContent(geminiModel, ...) // COMPILE ERROR
```

## 🚀 Quick Start

### Add Dependencies

```scala
// build.sbt
libraryDependencies ++= Seq(
  "io.github.iltotore" %% "iron" % "2.3.0",
  "org.typelevel" %% "cats-effect" % "3.5.2"
)
```

### Basic Usage

```scala
import cats.effect.*
import com.google.genai.scala.client.*
import com.google.genai.scala.types.*
import com.google.genai.scala.types.Model.*
import com.google.genai.scala.constraints.StringConstraints.*
import com.google.genai.scala.constraints.NumericConstraints.*

object Example extends IOApp.Simple:
  def run: IO[Unit] =
    GenAiClient.geminiFromEnv().use { client =>
      for
        // Prompt is validated: must be non-empty
        prompt <- IO.fromEither(
          Prompt("Explain quantum computing").left.map(new RuntimeException(_))
        )

        // Model type ensures text generation capability
        response <- client.generateContent(
          model = gemini20Flash,
          prompt = prompt
        )

        _ <- IO.println(response.firstText.getOrElse("No response"))
      yield ()
    }
```

## 📐 Iron Types in Action

### Numeric Constraints

```scala
import com.google.genai.scala.constraints.NumericConstraints.*

// Temperature: 0.0 - 2.0
val temp1: Either[String, Temperature] = Temperature(1.0)  // Right(1.0)
val temp2: Either[String, Temperature] = Temperature(3.0)  // Left("Should be less than or equal to 2.0")

// Compile-time validation with literals
val temp3: Temperature = Temperature.unsafe(1.5) // OK at compile-time
// val temp4: Temperature = Temperature.unsafe(5.0) // Runtime error

// TopP: 0.0 - 1.0 (nucleus sampling)
TopP(0.9)     // Right(0.9)
TopP(1.5)     // Left("Should be less than or equal to 1.0")

// TopK: > 0 (token selection)
TopK(40)      // Right(40)
TopK(-5)      // Left("Should be strictly greater than 0")

// MaxOutputTokens: > 0
MaxOutputTokens(1000)  // Right(1000)

// CandidateCount: 1-8
CandidateCount(3)      // Right(3)
CandidateCount(10)     // Left("Should be less than or equal to 8")

// AspectRatio: calculated from dimensions
AspectRatio(1920, 1080)  // Right(1.777...)
AspectRatio.Landscape_16_9  // Predefined ratios
AspectRatio.Square

// FileSizeBytes: 0 - 2GB with conversions
val size = FileSizeBytes.unsafe(1_073_741_824L)
size.toMB  // 1024.0
size.toGB  // 1.0
```

### String Constraints

```scala
import com.google.genai.scala.constraints.StringConstraints.*

// NonEmptyString: general purpose
NonEmptyString("hello")  // Right("hello")
NonEmptyString("")       // Left("Should be a non empty string")

// ApiKey: min length 20, with masking
ApiKey("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ")  // Right(...)
val key = ApiKey.unsafe("AIzaSy...")
key.masked  // "AIzaSy...4567"

// ProjectId: lowercase alphanumeric with hyphens, 6-30 chars
ProjectId("my-project-123")  // Right(...)
ProjectId("My-Project")      // Left("Should match [a-z][a-z0-9-]{4,28}[a-z0-9]")

// Location: Google Cloud regions
Location("us-central1")     // Right(...)
Location.USCentral1         // Predefined locations
Location.EuropeWest1

// ModelId: with predefined models
ModelId.Gemini_2_0_Flash
ModelId.Gemini_1_5_Pro
ModelId.Imagen_3_0
ModelId.Veo_2_0

// MimeType: with type detection
val mime = MimeType.ImagePng
mime.isImage  // true
mime.isVideo  // false
MimeType("video/mp4").map(_.isVideo)  // Right(true)

// Uri: with GCS/HTTP detection
val uri = Uri.unsafe("gs://my-bucket/file.txt")
uri.isGcs   // true
uri.isHttp  // false

// Base64Data: with encoding/decoding
val data = Base64Data.fromBytes("hello".getBytes)
data.decode  // Array[Byte]

// Prompt: non-empty with utilities
val prompt = Prompt.unsafe("Write a haiku")
prompt.wordCount  // 3
prompt.charCount  // 13
```

## 🎭 Phantom Types for API Variants

The type system prevents mixing Gemini API and Vertex AI operations:

```scala
// Gemini API (API key auth)
GenAiClient.gemini(apiKey).use { geminiClient =>
  // ✅ Available: text generation, image generation, file upload
  geminiClient.generateContent(...)
  geminiClient.uploadFile(...)

  // ❌ NOT available: tuning, file search stores
  // geminiClient.createTuningJob(...)  // Won't compile!
}

// Vertex AI (OAuth2/service account)
GenAiClient.vertex(projectId, location, token).use { vertexClient =>
  // ✅ Available: text generation, image generation, tuning
  vertexClient.generateContent(...)
  vertexClient.createTuningJob(...)

  // ❌ NOT available: file upload (use GCS instead)
  // vertexClient.uploadFile(...)  // Won't compile!
}
```

### How It Works

```scala
// Phantom type markers
sealed trait ApiVariant
sealed trait GeminiApi extends ApiVariant
sealed trait VertexApi extends ApiVariant

// Client is parameterized by API variant
trait GenAiClient[A <: ApiVariant]

// Gemini-specific operations
trait GeminiApiClient extends GenAiClient[GeminiApi]:
  def uploadFile(...): IO[FileId]  // Only available here

// Vertex-specific operations
trait VertexApiClient extends GenAiClient[VertexApi]:
  def createTuningJob(...): IO[TuningJobId]  // Only available here
```

## 🎯 Model Capabilities

Models are tagged with their capabilities, preventing invalid operations:

```scala
// Model with capabilities phantom types
case class Model[A <: ApiVariant, Caps <: ModelCapability]

// Predefined models with correct capabilities
val textModel: Model[GeminiApi, TextGeneration & FunctionCalling & CodeExecution] =
  Model.gemini20Flash

val imageModel: Model[GeminiApi, ImageGeneration] =
  Model.imagen3

val embeddingModel: Model[GeminiApi, Embeddings] =
  Model.textEmbedding

// Operations require specific capabilities
def generateContent[Caps](
  model: Model[A, Caps],
  prompt: Prompt
)(using HasCapability[Caps, TextGeneration]): IO[Response]

// ✅ This works: gemini20Flash has TextGeneration
client.generateContent(Model.gemini20Flash, prompt)

// ❌ This won't compile: imagen3 doesn't have TextGeneration
// client.generateContent(Model.imagen3, prompt)

// ✅ This works: imagen3 has ImageGeneration
client.generateImages(Model.imagen3, prompt)

// ❌ This won't compile: gemini20Flash doesn't have ImageGeneration
// client.generateImages(Model.gemini20Flash, prompt)
```

## 🛠️ Generation Configuration

All config parameters are iron-constrained:

```scala
// Predefined configs
GenerationConfig.creative      // High temp, diverse
GenerationConfig.deterministic // Low temp, focused
GenerationConfig.balanced      // Moderate settings

// Custom config with validation
val config = for
  temp <- Temperature(0.8)       // 0.0 - 2.0
  topP <- TopP(0.9)              // 0.0 - 1.0
  topK <- TopK(40)               // > 0
  maxTokens <- MaxOutputTokens(1000)
yield GenerationConfig(
  temperature = Some(temp),
  topP = Some(topP),
  topK = Some(topK),
  maxOutputTokens = Some(maxTokens)
)

config match
  case Right(cfg) => client.generateContent(model, prompt, cfg)
  case Left(err) => IO.println(s"Invalid config: $err")

// Or use unsafe for literals (validated at compile-time)
val quickConfig = GenerationConfig(
  temperature = Some(Temperature.unsafe(1.0)),
  maxOutputTokens = Some(MaxOutputTokens.unsafe(500))
)
```

## 🔐 Safety Settings

Type-safe safety configuration with ADTs:

```scala
// Predefined safety levels
SafetySetting.strictAll    // Block low and above
SafetySetting.permissive   // Block only high

// Custom safety settings
val customSafety = List(
  SafetySetting(HarmCategory.Harassment, HarmBlockThreshold.BlockMediumAndAbove),
  SafetySetting(HarmCategory.HateSpeech, HarmBlockThreshold.BlockLowAndAbove),
  SafetySetting(HarmCategory.DangerousContent, HarmBlockThreshold.BlockNone)
)

client.generateContent(
  model = gemini20Flash,
  prompt = prompt,
  safetySettings = customSafety
)
```

## 📦 Content Types (ADTs)

All content types use sealed traits for exhaustive pattern matching:

```scala
// Roles
sealed trait Role
case object User extends Role
case object Model extends Role
case object System extends Role

// Parts (content fragments)
sealed trait Part
case class Text(text: Prompt) extends Part
case class InlineData(data: Base64Data, mimeType: MimeType) extends Part
case class FileData(fileUri: Uri, mimeType: MimeType) extends Part
case class FunctionCall(name: NonEmptyString, args: Map[String, Any]) extends Part
// ... more part types

// Multi-part content
case class Content(role: Role, parts: List[Part])

// Pattern matching is exhaustive
def processPart(part: Part): String = part match
  case Text(text) => s"Text: ${text.value}"
  case InlineData(data, mime) => s"Data: ${mime.value}"
  case FileData(uri, mime) => s"File: ${uri.value}"
  case FunctionCall(name, _) => s"Function: ${name.value}"
  // Compiler ensures all cases handled!
```

## 🖼️ Image Generation

```scala
val imageConfig = ImageGenerationConfig(
  numberOfImages = CandidateCount.unsafe(2),
  aspectRatio = Some(AspectRatio.Landscape_16_9),
  negativePrompt = Some(Prompt.unsafe("blurry, distorted")),
  personGeneration = PersonGeneration.DontAllow
)

client.generateImages(
  model = Model.imagen3,
  prompt = Prompt.unsafe("Serene mountain landscape"),
  config = imageConfig
).flatMap { images =>
  images.traverse_ { image =>
    IO.println(s"Generated ${image.mimeType.value} image")
  }
}

// Predefined aspect ratios
AspectRatio.Square          // 1:1
AspectRatio.Landscape_16_9  // 16:9
AspectRatio.Portrait_9_16   // 9:16
AspectRatio.Landscape_4_3   // 4:3

// Custom aspect ratio
AspectRatio(1920, 1080)  // Calculate from dimensions
```

## 🎬 Video Generation

```scala
client.generateVideos(
  model = Model.veo2,
  prompt = Prompt.unsafe("A cat playing piano"),
  config = VideoGenerationConfig(
    aspectRatio = Some(AspectRatio.Landscape_16_9)
  )
)
```

## 📊 Embeddings

```scala
client.embedContent(
  model = Model.textEmbedding,
  content = Prompt.unsafe("machine learning"),
  taskType = Some(EmbeddingTaskType.SemanticSimilarity)
).map { response =>
  val embedding = response.embedding
  println(s"Dimension: ${embedding.dimension}")

  val normalized = embedding.normalize
  val similarity = embedding.cosineSimilarity(otherEmbedding)
  println(s"Similarity: $similarity")
}
```

## 📁 File Operations (Gemini API Only)

```scala
// Only available on GeminiApiClient
GenAiClient.gemini(apiKey).use { client =>
  for
    // Upload file
    fileId <- client.uploadFile(
      content = fileBytes,
      mimeType = MimeType.ImagePng,
      displayName = Some(NonEmptyString.unsafe("screenshot.png"))
    )

    // Get metadata
    metadata <- client.getFile(fileId)
    _ <- IO.println(s"Size: ${metadata.sizeBytes.toMB} MB")
    _ <- IO.println(s"Expires: ${metadata.expiresAt}")

    // List all files
    files <- client.listFiles
    _ <- files.traverse_(f => IO.println(f.name.value))

    // Delete file
    _ <- client.deleteFile(fileId)
  yield ()
}
```

## 🎓 Model Tuning (Vertex AI Only)

```scala
// Only available on VertexApiClient
GenAiClient.vertex(projectId, location, token).use { client =>
  for
    jobId <- client.createTuningJob(
      baseModel = ModelId.Gemini_1_5_Flash,
      trainingDataUri = Uri.unsafe("gs://my-bucket/training.jsonl"),
      config = TuningConfig(
        epochs = 10,
        learningRate = Some(0.001),
        batchSize = Some(32)
      )
    )

    status <- client.getTuningJob(jobId)
    _ <- status match
      case TuningJobStatus.Pending => IO.println("Pending...")
      case TuningJobStatus.Running => IO.println("Running...")
      case TuningJobStatus.Succeeded => IO.println("Done!")
      case TuningJobStatus.Failed => IO.println("Failed")
      case TuningJobStatus.Cancelled => IO.println("Cancelled")
  yield ()
}
```

## 📈 Token Usage & Costs

```scala
response.usageMetadata.foreach { usage =>
  println(s"Prompt tokens: ${usage.promptTokenCount}")
  println(s"Response tokens: ${usage.candidatesTokenCount}")
  println(s"Total tokens: ${usage.totalTokenCount}")

  // Calculate cost (example prices)
  val inputPricePerMillion = 0.075
  val outputPricePerMillion = 0.30
  val cost = usage.cost(inputPricePerMillion, outputPricePerMillion)
  println(f"Cost: $$${cost}%.6f")
}
```

## 🎯 Schema for Structured Output

```scala
// Define response schema
val personSchema = Schema.obj(
  properties = Map(
    "name" -> Schema.string("Person's full name"),
    "age" -> Schema.integer("Age in years", min = Some(0), max = Some(150)),
    "email" -> Schema.string("Email address"),
    "hobbies" -> Schema.array(Schema.string(), "List of hobbies")
  ),
  required = List("name", "age"),
  description = "A person object"
)

val config = GenerationConfig(
  responseSchema = Some(personSchema),
  responseMimeType = Some(MimeType.ApplicationJson)
)

client.generateContent(
  model = gemini20Flash,
  prompt = Prompt.unsafe("Generate a person profile"),
  config = config
)
```

## 🧠 Thinking Mode

```scala
val thinkingConfig = ThinkingConfig(
  thinkingBudget = ThinkingBudget.unsafe(4096)  // 1024-8192 tokens
)

client.generateContent(
  model = Model.gemini20FlashThinking,  // Must support ThinkingMode
  prompt = Prompt.unsafe("Solve this complex problem..."),
  thinkingConfig = Some(thinkingConfig)
)

// Response includes thought parts
response.candidates.foreach { candidate =>
  candidate.content.parts.foreach {
    case Part.Thought(thought) =>
      println(s"Model's reasoning: ${thought.value}")
    case Part.Text(text) =>
      println(s"Final answer: ${text.value}")
    case _ => ()
  }
}
```

## 🔧 Type Classes & Given Instances

The library uses Scala 3's given/using for capability checking:

```scala
// Capability type class
trait HasCapability[M <: ModelCapability, Cap <: ModelCapability]

// Automatic instances for capability checking
given reflexive[C <: ModelCapability]: HasCapability[C, C]
given intersectionLeft[C1, C2]: HasCapability[C1 & C2, C1]
given intersectionRight[C1, C2]: HasCapability[C1 & C2, C2]

// Used automatically in API methods
def generateContent[Caps](
  model: Model[A, Caps],
  prompt: Prompt
)(using HasCapability[Caps, TextGeneration]): IO[Response]
// Compiler finds appropriate given instance or fails
```

## 🎨 Benefits Summary

### Compile-Time Safety
- ✅ Can't mix Gemini/Vertex APIs
- ✅ Can't use wrong model for operation
- ✅ Can't forget required parameters (no nulls)
- ✅ Exhaustive pattern matching on ADTs

### Runtime Validation
- ✅ All numeric ranges validated (temperature, topP, etc.)
- ✅ All string formats validated (API keys, project IDs, MIME types)
- ✅ Clear error messages for validation failures
- ✅ No primitive obsession (every value has semantic type)

### Type-Level Features
- ✅ Phantom types for API variants
- ✅ Capability types for model features
- ✅ Opaque types for zero-cost abstractions
- ✅ Iron refinements for constraints

### Developer Experience
- ✅ Autocomplete shows only valid operations
- ✅ Impossible states are impossible to represent
- ✅ Refactoring is safe (compiler catches breaks)
- ✅ Documentation in types (self-documenting API)

## 📚 Further Reading

- [Iron Library](https://github.com/Iltotore/iron) - Refinement types for Scala 3
- [Phantom Types](https://blog.rockthejvm.com/phantom-types/) - Compile-time constraints
- [Opaque Types](https://docs.scala-lang.org/scala3/reference/other-new-features/opaques.html) - Scala 3 feature
- [Google GenAI Java SDK](https://github.com/googleapis/java-genai) - Underlying implementation

## 📄 License

Apache 2.0 (same as Google GenAI Java SDK)
