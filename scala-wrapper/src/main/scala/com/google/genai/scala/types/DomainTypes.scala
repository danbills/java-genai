package com.google.genai.scala.types

import com.google.genai.scala.constraints.NumericConstraints.*
import com.google.genai.scala.constraints.StringConstraints.*

/**
 * Core domain types using ADTs (Algebraic Data Types).
 * Sealed traits ensure exhaustive pattern matching at compile-time.
 */

// ============================================================================
// ROLES & CONTENT
// ============================================================================

/** Role in a conversation */
sealed trait Role derives CanEqual
object Role:
  case object User extends Role
  case object Model extends Role
  case object System extends Role

/** Content part types using ADT */
sealed trait Part derives CanEqual
object Part:
  case class Text(text: Prompt) extends Part
  case class InlineData(data: Base64Data, mimeType: MimeType) extends Part
  case class FileData(fileUri: Uri, mimeType: MimeType) extends Part
  case class FunctionCall(name: NonEmptyString, args: Map[String, Any]) extends Part
  case class FunctionResponse(name: NonEmptyString, response: Map[String, Any]) extends Part
  case class ExecutableCode(language: CodeLanguage, code: NonEmptyString) extends Part
  case class CodeExecutionResult(outcome: ExecutionOutcome, output: String) extends Part
  case class Thought(text: NonEmptyString) extends Part

/** Programming languages for executable code */
sealed trait CodeLanguage derives CanEqual
object CodeLanguage:
  case object Python extends CodeLanguage
  case object JavaScript extends CodeLanguage

/** Code execution outcome */
sealed trait ExecutionOutcome derives CanEqual
object ExecutionOutcome:
  case object Success extends ExecutionOutcome
  case object Failure extends ExecutionOutcome

/** Multi-part content with role */
case class Content(
  role: Role,
  parts: List[Part]
) derives CanEqual:
  require(parts.nonEmpty, "Content must have at least one part")

// ============================================================================
// GENERATION CONFIG
// ============================================================================

/** Generation configuration with ultra-constrained types */
case class GenerationConfig(
  temperature: Option[Temperature] = None,
  topP: Option[TopP] = None,
  topK: Option[TopK] = None,
  maxOutputTokens: Option[MaxOutputTokens] = None,
  candidateCount: Option[CandidateCount] = None,
  stopSequences: List[NonEmptyString] = Nil,
  presencePenalty: Option[Double] = None,
  frequencyPenalty: Option[Double] = None,
  responseSchema: Option[Schema] = None,
  responseMimeType: Option[MimeType] = None
) derives CanEqual

object GenerationConfig:
  /** Creative config: high temperature, diverse outputs */
  def creative: GenerationConfig = GenerationConfig(
    temperature = Some(Temperature.unsafe(1.5)),
    topP = Some(TopP.unsafe(0.95)),
    topK = Some(TopK.unsafe(40))
  )

  /** Deterministic config: low temperature, focused outputs */
  def deterministic: GenerationConfig = GenerationConfig(
    temperature = Some(Temperature.unsafe(0.0)),
    topP = Some(TopP.unsafe(0.1)),
    topK = Some(TopK.unsafe(1))
  )

  /** Balanced config: moderate settings */
  def balanced: GenerationConfig = GenerationConfig(
    temperature = Some(Temperature.unsafe(0.7)),
    topP = Some(TopP.unsafe(0.8)),
    topK = Some(TopK.unsafe(20))
  )

// ============================================================================
// SAFETY SETTINGS
// ============================================================================

/** Harm categories for content safety */
sealed trait HarmCategory derives CanEqual
object HarmCategory:
  case object Harassment extends HarmCategory
  case object HateSpeech extends HarmCategory
  case object SexuallyExplicit extends HarmCategory
  case object DangerousContent extends HarmCategory
  case object CivicIntegrity extends HarmCategory

/** Harm block threshold levels */
sealed trait HarmBlockThreshold derives CanEqual
object HarmBlockThreshold:
  case object BlockNone extends HarmBlockThreshold
  case object BlockLowAndAbove extends HarmBlockThreshold
  case object BlockMediumAndAbove extends HarmBlockThreshold
  case object BlockOnlyHigh extends HarmBlockThreshold

/** Safety setting for a specific harm category */
case class SafetySetting(
  category: HarmCategory,
  threshold: HarmBlockThreshold
) derives CanEqual

object SafetySetting:
  /** Block all harmful content */
  def strictAll: List[SafetySetting] = List(
    SafetySetting(HarmCategory.Harassment, HarmBlockThreshold.BlockLowAndAbove),
    SafetySetting(HarmCategory.HateSpeech, HarmBlockThreshold.BlockLowAndAbove),
    SafetySetting(HarmCategory.SexuallyExplicit, HarmBlockThreshold.BlockLowAndAbove),
    SafetySetting(HarmCategory.DangerousContent, HarmBlockThreshold.BlockLowAndAbove),
    SafetySetting(HarmCategory.CivicIntegrity, HarmBlockThreshold.BlockLowAndAbove)
  )

  /** Permissive settings */
  def permissive: List[SafetySetting] = List(
    SafetySetting(HarmCategory.Harassment, HarmBlockThreshold.BlockOnlyHigh),
    SafetySetting(HarmCategory.HateSpeech, HarmBlockThreshold.BlockOnlyHigh),
    SafetySetting(HarmCategory.SexuallyExplicit, HarmBlockThreshold.BlockOnlyHigh),
    SafetySetting(HarmCategory.DangerousContent, HarmBlockThreshold.BlockOnlyHigh),
    SafetySetting(HarmCategory.CivicIntegrity, HarmBlockThreshold.BlockOnlyHigh)
  )

// ============================================================================
// RESPONSE TYPES
// ============================================================================

/** Finish reason for generation */
sealed trait FinishReason derives CanEqual
object FinishReason:
  case object Stop extends FinishReason
  case object MaxTokens extends FinishReason
  case object Safety extends FinishReason
  case object Recitation extends FinishReason
  case object Other extends FinishReason
  case object Blocklist extends FinishReason
  case object ProhibitedContent extends FinishReason
  case object Spii extends FinishReason

/** Safety rating */
case class SafetyRating(
  category: HarmCategory,
  probability: HarmProbability,
  blocked: Boolean
) derives CanEqual

/** Harm probability levels */
sealed trait HarmProbability derives CanEqual
object HarmProbability:
  case object Negligible extends HarmProbability
  case object Low extends HarmProbability
  case object Medium extends HarmProbability
  case object High extends HarmProbability

/** Response candidate */
case class Candidate(
  content: Content,
  finishReason: Option[FinishReason],
  safetyRatings: List[SafetyRating],
  index: Int
) derives CanEqual

/** Token usage metadata */
case class UsageMetadata(
  promptTokenCount: Int,
  candidatesTokenCount: Int,
  totalTokenCount: Int
) derives CanEqual:
  def cost(inputPricePerMillion: Double, outputPricePerMillion: Double): Double =
    (promptTokenCount * inputPricePerMillion + candidatesTokenCount * outputPricePerMillion) / 1_000_000.0

/** Complete generation response */
case class GenerateContentResponse(
  candidates: List[Candidate],
  usageMetadata: Option[UsageMetadata],
  modelVersion: Option[String]
) derives CanEqual:
  def firstText: Option[String] = candidates.headOption.flatMap { candidate =>
    candidate.content.parts.collectFirst {
      case Part.Text(text) => text.value
    }
  }

  def allText: String = candidates.flatMap { candidate =>
    candidate.content.parts.collect {
      case Part.Text(text) => text.value
    }
  }.mkString("\n")

// ============================================================================
// SCHEMA TYPES (for structured outputs)
// ============================================================================

/** JSON Schema types */
sealed trait SchemaType derives CanEqual
object SchemaType:
  case object String extends SchemaType
  case object Number extends SchemaType
  case object Integer extends SchemaType
  case object Boolean extends SchemaType
  case object Array extends SchemaType
  case object Object extends SchemaType

/** JSON Schema for structured outputs */
case class Schema(
  `type`: SchemaType,
  description: Option[NonEmptyString] = None,
  properties: Map[String, Schema] = Map.empty,
  required: List[NonEmptyString] = Nil,
  items: Option[Schema] = None,
  enum: List[String] = Nil,
  format: Option[String] = None,
  nullable: Boolean = false,
  minimum: Option[Double] = None,
  maximum: Option[Double] = None,
  minLength: Option[Int] = None,
  maxLength: Option[Int] = None,
  pattern: Option[String] = None
) derives CanEqual

object Schema:
  def string(description: String = ""): Schema =
    Schema(SchemaType.String, Option.when(description.nonEmpty)(NonEmptyString.unsafe(description)))

  def integer(description: String = "", min: Option[Int] = None, max: Option[Int] = None): Schema =
    Schema(
      SchemaType.Integer,
      Option.when(description.nonEmpty)(NonEmptyString.unsafe(description)),
      minimum = min.map(_.toDouble),
      maximum = max.map(_.toDouble)
    )

  def boolean(description: String = ""): Schema =
    Schema(SchemaType.Boolean, Option.when(description.nonEmpty)(NonEmptyString.unsafe(description)))

  def array(itemSchema: Schema, description: String = ""): Schema =
    Schema(
      SchemaType.Array,
      Option.when(description.nonEmpty)(NonEmptyString.unsafe(description)),
      items = Some(itemSchema)
    )

  def obj(
    properties: Map[String, Schema],
    required: List[String] = Nil,
    description: String = ""
  ): Schema =
    Schema(
      SchemaType.Object,
      Option.when(description.nonEmpty)(NonEmptyString.unsafe(description)),
      properties = properties,
      required = required.map(NonEmptyString.unsafe)
    )

// ============================================================================
// TOOL TYPES
// ============================================================================

/** Tool definition */
sealed trait Tool derives CanEqual
object Tool:
  case class FunctionDeclaration(
    name: NonEmptyString,
    description: NonEmptyString,
    parameters: Schema
  ) extends Tool

  case object CodeExecution extends Tool
  case object GoogleSearch extends Tool

/** Tool configuration */
case class ToolConfig(
  functionCallingMode: FunctionCallingMode
) derives CanEqual

/** Function calling modes */
sealed trait FunctionCallingMode derives CanEqual
object FunctionCallingMode:
  case object Auto extends FunctionCallingMode
  case object Any extends FunctionCallingMode
  case object None extends FunctionCallingMode

// ============================================================================
// THINKING MODE
// ============================================================================

/** Thinking configuration for reasoning models */
case class ThinkingConfig(
  thinkingBudget: ThinkingBudget
) derives CanEqual

// ============================================================================
// IMAGE GENERATION
// ============================================================================

/** Image generation configuration */
case class ImageGenerationConfig(
  numberOfImages: CandidateCount = CandidateCount.default,
  aspectRatio: Option[AspectRatio] = None,
  negativePrompt: Option[Prompt] = None,
  safetySettings: List[SafetySetting] = SafetySetting.permissive,
  personGeneration: PersonGeneration = PersonGeneration.DontAllow
) derives CanEqual

/** Person generation control */
sealed trait PersonGeneration derives CanEqual
object PersonGeneration:
  case object Allow extends PersonGeneration
  case object DontAllow extends PersonGeneration

/** Generated image */
case class GeneratedImage(
  data: Base64Data,
  mimeType: MimeType
) derives CanEqual

// ============================================================================
// VIDEO GENERATION
// ============================================================================

/** Video generation configuration */
case class VideoGenerationConfig(
  aspectRatio: Option[AspectRatio] = None,
  negativePrompt: Option[Prompt] = None
) derives CanEqual

// ============================================================================
// EMBEDDINGS
// ============================================================================

/** Embedding task type */
sealed trait EmbeddingTaskType derives CanEqual
object EmbeddingTaskType:
  case object RetrievalQuery extends EmbeddingTaskType
  case object RetrievalDocument extends EmbeddingTaskType
  case object SemanticSimilarity extends EmbeddingTaskType
  case object Classification extends EmbeddingTaskType
  case object Clustering extends EmbeddingTaskType

/** Embedding vector (dimensionality enforced at runtime) */
case class Embedding(
  values: Vector[Double]
) derives CanEqual:
  def dimension: Int = values.length
  def normalize: Embedding = {
    val magnitude = math.sqrt(values.map(v => v * v).sum)
    Embedding(values.map(_ / magnitude))
  }
  def cosineSimilarity(other: Embedding): Double = {
    require(dimension == other.dimension, "Embeddings must have same dimension")
    values.zip(other.values).map { case (a, b) => a * b }.sum
  }

/** Embedding response */
case class EmbedContentResponse(
  embedding: Embedding
) derives CanEqual
