package com.google.genai.scala.types

import com.google.genai.scala.constraints.StringConstraints.*

/**
 * Phantom types for compile-time API variant enforcement.
 * Prevents calling Vertex AI operations on Gemini API clients and vice versa.
 */

/** Phantom type marker for API variants */
sealed trait ApiVariant

/** Gemini Developer API (API key authentication) */
sealed trait GeminiApi extends ApiVariant

/** Vertex AI API (OAuth2/service account authentication) */
sealed trait VertexApi extends ApiVariant

/** Operations available in both APIs */
sealed trait BothApis extends GeminiApi with VertexApi

/**
 * Phantom-typed credentials.
 * Type parameter ensures correct credential type for API variant.
 */
sealed trait Credentials[+A <: ApiVariant]

object Credentials:
  /** API key for Gemini Developer API */
  case class ApiKeyCredentials(apiKey: ApiKey) extends Credentials[GeminiApi]

  /** OAuth2 credentials for Vertex AI */
  case class OAuth2Credentials(
    projectId: ProjectId,
    location: Location,
    accessToken: NonEmptyString
  ) extends Credentials[VertexApi]

  /** Service account credentials for Vertex AI */
  case class ServiceAccountCredentials(
    projectId: ProjectId,
    location: Location,
    credentialsPath: NonEmptyString
  ) extends Credentials[VertexApi]

/**
 * Capability markers for model features.
 * Enables compile-time checking of model capabilities.
 */
sealed trait ModelCapability

/** Model supports text generation */
sealed trait TextGeneration extends ModelCapability

/** Model supports image generation */
sealed trait ImageGeneration extends ModelCapability

/** Model supports video generation */
sealed trait VideoGeneration extends ModelCapability

/** Model supports embeddings */
sealed trait Embeddings extends ModelCapability

/** Model supports function calling */
sealed trait FunctionCalling extends ModelCapability

/** Model supports code execution */
sealed trait CodeExecution extends ModelCapability

/** Model supports thinking/reasoning */
sealed trait ThinkingMode extends ModelCapability

/**
 * Phantom-typed model reference.
 * Type parameters encode both API variant and capabilities.
 */
case class Model[A <: ApiVariant, Caps <: ModelCapability](
  id: ModelId
)

object Model:
  /** Gemini 2.0 Flash (supports text, function calling, code execution) */
  type Gemini20Flash[A <: ApiVariant] = Model[A, TextGeneration & FunctionCalling & CodeExecution]

  def gemini20Flash[A <: ApiVariant]: Model[A, TextGeneration & FunctionCalling & CodeExecution] =
    Model(ModelId.Gemini_2_0_Flash)

  /** Gemini 2.0 Flash Thinking (adds thinking mode) */
  type Gemini20FlashThinking[A <: ApiVariant] =
    Model[A, TextGeneration & FunctionCalling & CodeExecution & ThinkingMode]

  def gemini20FlashThinking[A <: ApiVariant]:
    Model[A, TextGeneration & FunctionCalling & CodeExecution & ThinkingMode] =
    Model(ModelId.Gemini_2_0_Flash_Thinking)

  /** Gemini 1.5 Pro */
  type Gemini15Pro[A <: ApiVariant] = Model[A, TextGeneration & FunctionCalling & CodeExecution]

  def gemini15Pro[A <: ApiVariant]: Model[A, TextGeneration & FunctionCalling & CodeExecution] =
    Model(ModelId.Gemini_1_5_Pro)

  /** Imagen 3.0 (image generation only) */
  type Imagen3[A <: ApiVariant] = Model[A, ImageGeneration]

  def imagen3[A <: ApiVariant]: Model[A, ImageGeneration] =
    Model(ModelId.Imagen_3_0)

  /** Veo 2.0 (video generation only) */
  type Veo2[A <: ApiVariant] = Model[A, VideoGeneration]

  def veo2[A <: ApiVariant]: Model[A, VideoGeneration] =
    Model(ModelId.Veo_2_0)

  /** Text Embedding model */
  type TextEmbedding[A <: ApiVariant] = Model[A, Embeddings]

  def textEmbedding[A <: ApiVariant]: Model[A, Embeddings] =
    Model(ModelId.TextEmbedding)

/**
 * Feature availability markers.
 * Some features are only available in specific API variants.
 */
sealed trait Feature[+A <: ApiVariant]

object Feature:
  /** File API (Gemini only) */
  case object Files extends Feature[GeminiApi]

  /** Cached content (both APIs) */
  case object Caches extends Feature[BothApis]

  /** Batch operations (both APIs) */
  case object Batches extends Feature[BothApis]

  /** Model tuning (Vertex AI only) */
  case object Tuning extends Feature[VertexApi]

  /** File search stores (Vertex AI only) */
  case object FileSearchStores extends Feature[VertexApi]

/**
 * Type-level proof that a feature is available for an API variant.
 * This enables compile-time enforcement of feature availability.
 */
trait FeatureAvailable[F <: Feature[A], A <: ApiVariant]

object FeatureAvailable:
  // Files only available in Gemini API
  given filesInGemini: FeatureAvailable[Feature.Files.type, GeminiApi] = new FeatureAvailable {}

  // Caches available in both APIs
  given cachesInGemini: FeatureAvailable[Feature.Caches.type, GeminiApi] = new FeatureAvailable {}
  given cachesInVertex: FeatureAvailable[Feature.Caches.type, VertexApi] = new FeatureAvailable {}

  // Batches available in both APIs
  given batchesInGemini: FeatureAvailable[Feature.Batches.type, GeminiApi] = new FeatureAvailable {}
  given batchesInVertex: FeatureAvailable[Feature.Batches.type, VertexApi] = new FeatureAvailable {}

  // Tuning only available in Vertex AI
  given tuningInVertex: FeatureAvailable[Feature.Tuning.type, VertexApi] = new FeatureAvailable {}

  // File search stores only in Vertex AI
  given fileSearchInVertex: FeatureAvailable[Feature.FileSearchStores.type, VertexApi] =
    new FeatureAvailable {}

/**
 * Type-level proof that a model has a capability.
 * Enables compile-time enforcement of operation availability.
 */
trait HasCapability[M <: ModelCapability, Cap <: ModelCapability]

object HasCapability:
  // Reflexive: a capability has itself
  given reflexive[C <: ModelCapability]: HasCapability[C, C] = new HasCapability {}

  // Intersection left: C1 & C2 has C1
  given intersectionLeft[C1 <: ModelCapability, C2 <: ModelCapability]:
    HasCapability[C1 & C2, C1] = new HasCapability {}

  // Intersection right: C1 & C2 has C2
  given intersectionRight[C1 <: ModelCapability, C2 <: ModelCapability]:
    HasCapability[C1 & C2, C2] = new HasCapability {}

  // Transitive chaining for deeper intersections
  given transitive[M <: ModelCapability, C1 <: ModelCapability, C2 <: ModelCapability](
    using ev1: HasCapability[M, C1],
    ev2: HasCapability[C1, C2]
  ): HasCapability[M, C2] = new HasCapability {}
