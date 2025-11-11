package com.google.genai.scala.client

import cats.effect.*
import cats.syntax.all.*
import com.google.genai.scala.types.*
import com.google.genai.scala.types.PhantomTypes.*
import com.google.genai.scala.constraints.StringConstraints.*

/**
 * Ultra-constrained GenAI client with phantom types.
 *
 * Type parameters ensure:
 * - A: API variant (GeminiApi or VertexApi) - prevents mixing API calls
 * - Only operations compatible with the API variant are available
 *
 * Uses iron types throughout for:
 * - Compile-time validation of literals
 * - Runtime validation with descriptive errors
 * - Complete elimination of primitive obsession
 */
trait GenAiClient[A <: ApiVariant]:

  /**
   * Generate text content.
   * Only available for models with TextGeneration capability.
   */
  def generateContent[Caps <: ModelCapability](
    model: Model[A, Caps],
    prompt: Prompt,
    config: GenerationConfig = GenerationConfig(),
    safetySettings: List[SafetySetting] = Nil,
    tools: List[Tool] = Nil
  )(using HasCapability[Caps, TextGeneration]): IO[GenerateContentResponse]

  /**
   * Generate text content with conversation history.
   */
  def generateContentWithHistory[Caps <: ModelCapability](
    model: Model[A, Caps],
    history: List[Content],
    config: GenerationConfig = GenerationConfig(),
    safetySettings: List[SafetySetting] = Nil
  )(using HasCapability[Caps, TextGeneration]): IO[GenerateContentResponse]

  /**
   * Generate images.
   * Only available for models with ImageGeneration capability.
   */
  def generateImages[Caps <: ModelCapability](
    model: Model[A, Caps],
    prompt: Prompt,
    config: ImageGenerationConfig = ImageGenerationConfig()
  )(using HasCapability[Caps, ImageGeneration]): IO[List[GeneratedImage]]

  /**
   * Generate embeddings.
   * Only available for embedding models.
   */
  def embedContent[Caps <: ModelCapability](
    model: Model[A, Caps],
    content: Prompt,
    taskType: Option[EmbeddingTaskType] = None
  )(using HasCapability[Caps, Embeddings]): IO[EmbedContentResponse]

  /**
   * Count tokens in content.
   * Available for any text-capable model.
   */
  def countTokens[Caps <: ModelCapability](
    model: Model[A, Caps],
    content: Prompt
  )(using HasCapability[Caps, TextGeneration]): IO[Int]

/**
 * Gemini API client (API key authentication).
 * Provides file upload/management operations only available in Gemini API.
 */
trait GeminiApiClient extends GenAiClient[GeminiApi]:

  /**
   * Upload a file (Gemini API only).
   * Returns FileId for use in subsequent requests.
   */
  def uploadFile(
    content: Array[Byte],
    mimeType: MimeType,
    displayName: Option[NonEmptyString] = None
  ): IO[FileId]

  /**
   * Get file metadata.
   */
  def getFile(fileId: FileId): IO[FileMetadata]

  /**
   * Delete a file.
   */
  def deleteFile(fileId: FileId): IO[Unit]

  /**
   * List all uploaded files.
   */
  def listFiles: IO[List[FileMetadata]]

/**
 * Vertex AI client (OAuth2/service account authentication).
 * Provides tuning and file search operations only available in Vertex AI.
 */
trait VertexApiClient extends GenAiClient[VertexApi]:

  /**
   * Create a tuning job (Vertex AI only).
   */
  def createTuningJob(
    baseModel: ModelId,
    trainingDataUri: Uri,
    config: TuningConfig
  ): IO[TuningJobId]

  /**
   * Get tuning job status.
   */
  def getTuningJob(jobId: TuningJobId): IO[TuningJobStatus]

/**
 * File metadata from file upload.
 */
case class FileMetadata(
  id: FileId,
  name: NonEmptyString,
  mimeType: MimeType,
  sizeBytes: FileSizeBytes,
  createdAt: java.time.Instant,
  expiresAt: java.time.Instant
) derives CanEqual

/**
 * Tuning job identifier.
 */
opaque type TuningJobId = String

object TuningJobId:
  def apply(value: String): Either[String, TuningJobId] =
    if value.isEmpty then Left("TuningJobId cannot be empty")
    else Right(value)

  def unsafe(value: String): TuningJobId = value

  extension (t: TuningJobId)
    def value: String = t

/**
 * Tuning configuration.
 */
case class TuningConfig(
  epochs: Int,
  learningRate: Option[Double] = None,
  batchSize: Option[Int] = None
) derives CanEqual

/**
 * Tuning job status.
 */
sealed trait TuningJobStatus derives CanEqual
object TuningJobStatus:
  case object Pending extends TuningJobStatus
  case object Running extends TuningJobStatus
  case object Succeeded extends TuningJobStatus
  case object Failed extends TuningJobStatus
  case object Cancelled extends TuningJobStatus

/**
 * Client factory with smart constructors that enforce credential types.
 */
object GenAiClient:

  /**
   * Create Gemini API client with API key.
   * Type signature ensures only ApiKeyCredentials can be used.
   */
  def gemini(apiKey: ApiKey): Resource[IO, GeminiApiClient] =
    Resource.pure(new GeminiApiClientImpl(apiKey))

  /**
   * Create Gemini API client from environment variable.
   */
  def geminiFromEnv(envVar: String = "GEMINI_API_KEY"): Resource[IO, GeminiApiClient] =
    Resource.eval(
      IO.fromEither(
        ApiKey.fromEnv(envVar).left.map(err => new RuntimeException(err))
      )
    ).flatMap(gemini)

  /**
   * Create Vertex AI client with OAuth2 credentials.
   * Type signature ensures only OAuth2/ServiceAccount credentials can be used.
   */
  def vertex(
    projectId: ProjectId,
    location: Location,
    accessToken: NonEmptyString
  ): Resource[IO, VertexApiClient] =
    Resource.pure(new VertexApiClientImpl(projectId, location, accessToken))

  /**
   * Create Vertex AI client with service account.
   */
  def vertexWithServiceAccount(
    projectId: ProjectId,
    location: Location,
    credentialsPath: NonEmptyString
  ): Resource[IO, VertexApiClient] =
    Resource.pure(new VertexApiClientImpl(projectId, location, credentialsPath))

/**
 * Internal implementation for Gemini API.
 * Not exposed directly - users must use GenAiClient.gemini factory.
 */
private class GeminiApiClientImpl(apiKey: ApiKey) extends GeminiApiClient:

  def generateContent[Caps <: ModelCapability](
    model: Model[GeminiApi, Caps],
    prompt: Prompt,
    config: GenerationConfig,
    safetySettings: List[SafetySetting],
    tools: List[Tool]
  )(using HasCapability[Caps, TextGeneration]): IO[GenerateContentResponse] =
    // Implementation would call underlying Java SDK
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def generateContentWithHistory[Caps <: ModelCapability](
    model: Model[GeminiApi, Caps],
    history: List[Content],
    config: GenerationConfig,
    safetySettings: List[SafetySetting]
  )(using HasCapability[Caps, TextGeneration]): IO[GenerateContentResponse] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def generateImages[Caps <: ModelCapability](
    model: Model[GeminiApi, Caps],
    prompt: Prompt,
    config: ImageGenerationConfig
  )(using HasCapability[Caps, ImageGeneration]): IO[List[GeneratedImage]] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def embedContent[Caps <: ModelCapability](
    model: Model[GeminiApi, Caps],
    content: Prompt,
    taskType: Option[EmbeddingTaskType]
  )(using HasCapability[Caps, Embeddings]): IO[EmbedContentResponse] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def countTokens[Caps <: ModelCapability](
    model: Model[GeminiApi, Caps],
    content: Prompt
  )(using HasCapability[Caps, TextGeneration]): IO[Int] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def uploadFile(
    content: Array[Byte],
    mimeType: MimeType,
    displayName: Option[NonEmptyString]
  ): IO[FileId] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def getFile(fileId: FileId): IO[FileMetadata] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def deleteFile(fileId: FileId): IO[Unit] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def listFiles: IO[List[FileMetadata]] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

/**
 * Internal implementation for Vertex AI.
 */
private class VertexApiClientImpl(
  projectId: ProjectId,
  location: Location,
  credentials: NonEmptyString
) extends VertexApiClient:

  def generateContent[Caps <: ModelCapability](
    model: Model[VertexApi, Caps],
    prompt: Prompt,
    config: GenerationConfig,
    safetySettings: List[SafetySetting],
    tools: List[Tool]
  )(using HasCapability[Caps, TextGeneration]): IO[GenerateContentResponse] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def generateContentWithHistory[Caps <: ModelCapability](
    model: Model[VertexApi, Caps],
    history: List[Content],
    config: GenerationConfig,
    safetySettings: List[SafetySetting]
  )(using HasCapability[Caps, TextGeneration]): IO[GenerateContentResponse] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def generateImages[Caps <: ModelCapability](
    model: Model[VertexApi, Caps],
    prompt: Prompt,
    config: ImageGenerationConfig
  )(using HasCapability[Caps, ImageGeneration]): IO[List[GeneratedImage]] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def embedContent[Caps <: ModelCapability](
    model: Model[VertexApi, Caps],
    content: Prompt,
    taskType: Option[EmbeddingTaskType]
  )(using HasCapability[Caps, Embeddings]): IO[EmbedContentResponse] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def countTokens[Caps <: ModelCapability](
    model: Model[VertexApi, Caps],
    content: Prompt
  )(using HasCapability[Caps, TextGeneration]): IO[Int] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def createTuningJob(
    baseModel: ModelId,
    trainingDataUri: Uri,
    config: TuningConfig
  ): IO[TuningJobId] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))

  def getTuningJob(jobId: TuningJobId): IO[TuningJobStatus] =
    IO.raiseError(new NotImplementedError("Implementation bridges to Java SDK"))
