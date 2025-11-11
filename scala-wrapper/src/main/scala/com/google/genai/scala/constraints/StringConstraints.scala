package com.google.genai.scala.constraints

import io.github.iltotore.iron.*
import io.github.iltotore.iron.constraint.all.*

/**
 * Ultra-constrained string types using Iron refinement types.
 * Prevents primitive obsession and enforces domain constraints.
 */
object StringConstraints:

  // NonEmptyString: General purpose non-empty string
  type NonEmptyStringConstraint = Not[Empty]
  opaque type NonEmptyString = String :| NonEmptyStringConstraint

  object NonEmptyString:
    def apply(value: String): Either[String, NonEmptyString] =
      value.refineEither[NonEmptyStringConstraint]

    def unsafe(value: String): NonEmptyString =
      value.refineUnsafe[NonEmptyStringConstraint]

    extension (s: NonEmptyString)
      def value: String = s

  // ApiKey: Google API key (must be non-empty, typically starts with "AIza")
  type ApiKeyConstraint = Not[Empty] & MinLength[20]
  opaque type ApiKey = String :| ApiKeyConstraint

  object ApiKey:
    def apply(value: String): Either[String, ApiKey] =
      value.refineEither[ApiKeyConstraint]

    def unsafe(value: String): ApiKey =
      value.refineUnsafe[ApiKeyConstraint]

    def fromEnv(key: String = "GEMINI_API_KEY"): Either[String, ApiKey] =
      Option(System.getenv(key))
        .toRight(s"Environment variable $key not found")
        .flatMap(apply)

    extension (a: ApiKey)
      def value: String = a
      def masked: String = s"${a.take(10)}...${a.takeRight(4)}"

  // ProjectId: Google Cloud project ID (lowercase alphanumeric with hyphens)
  type ProjectIdConstraint = Not[Empty] & Match["[a-z][a-z0-9-]{4,28}[a-z0-9]"]
  opaque type ProjectId = String :| ProjectIdConstraint

  object ProjectId:
    def apply(value: String): Either[String, ProjectId] =
      value.refineEither[ProjectIdConstraint]

    def unsafe(value: String): ProjectId =
      value.refineUnsafe[ProjectIdConstraint]

    extension (p: ProjectId)
      def value: String = p

  // Location: Google Cloud location (e.g., "us-central1", "europe-west1")
  type LocationConstraint = Not[Empty] & Match["[a-z]+-[a-z]+[0-9]"]
  opaque type Location = String :| LocationConstraint

  object Location:
    def apply(value: String): Either[String, Location] =
      value.refineEither[LocationConstraint]

    def unsafe(value: String): Location =
      value.refineUnsafe[LocationConstraint]

    // Common locations
    val USCentral1: Location = unsafe("us-central1")
    val USEast1: Location = unsafe("us-east1")
    val USWest1: Location = unsafe("us-west1")
    val EuropeWest1: Location = unsafe("europe-west1")
    val EuropeWest4: Location = unsafe("europe-west4")
    val AsiaSoutheast1: Location = unsafe("asia-southeast1")

    extension (l: Location)
      def value: String = l

  // ModelId: Model identifier (e.g., "gemini-2.0-flash", "models/gemini-pro")
  opaque type ModelId = String :| Not[Empty]

  object ModelId:
    def apply(value: String): Either[String, ModelId] =
      value.refineEither[Not[Empty]]

    def unsafe(value: String): ModelId =
      value.refineUnsafe[Not[Empty]]

    // Well-known model IDs as literal types
    val Gemini_2_0_Flash: ModelId = unsafe("gemini-2.0-flash-exp")
    val Gemini_2_0_Flash_Thinking: ModelId = unsafe("gemini-2.0-flash-thinking-exp")
    val Gemini_1_5_Pro: ModelId = unsafe("gemini-1.5-pro")
    val Gemini_1_5_Flash: ModelId = unsafe("gemini-1.5-flash")
    val Gemini_1_5_Flash_8B: ModelId = unsafe("gemini-1.5-flash-8b")
    val Imagen_3_0: ModelId = unsafe("imagen-3.0-generate-001")
    val Imagen_3_0_Fast: ModelId = unsafe("imagen-3.0-fast-generate-001")
    val Veo_2_0: ModelId = unsafe("veo-2.0-generate-001")
    val TextEmbedding: ModelId = unsafe("text-embedding-004")

    extension (m: ModelId)
      def value: String = m
      def withPrefix: String = if m.startsWith("models/") then m else s"models/$m"

  // FileId: File resource identifier
  opaque type FileId = String :| Not[Empty]

  object FileId:
    def apply(value: String): Either[String, FileId] =
      value.refineEither[Not[Empty]]

    def unsafe(value: String): FileId =
      value.refineUnsafe[Not[Empty]]

    extension (f: FileId)
      def value: String = f

  // MimeType: MIME type for files
  type MimeTypeConstraint = Not[Empty] & Match["[a-z]+/[a-z0-9.+-]+"]
  opaque type MimeType = String :| MimeTypeConstraint

  object MimeType:
    def apply(value: String): Either[String, MimeType] =
      value.refineEither[MimeTypeConstraint]

    def unsafe(value: String): MimeType =
      value.refineUnsafe[MimeTypeConstraint]

    // Common MIME types
    val TextPlain: MimeType = unsafe("text/plain")
    val ApplicationJson: MimeType = unsafe("application/json")
    val ImagePng: MimeType = unsafe("image/png")
    val ImageJpeg: MimeType = unsafe("image/jpeg")
    val ImageWebp: MimeType = unsafe("image/webp")
    val VideoMp4: MimeType = unsafe("video/mp4")
    val VideoMov: MimeType = unsafe("video/mov")
    val AudioMp3: MimeType = unsafe("audio/mp3")
    val AudioWav: MimeType = unsafe("audio/wav")
    val ApplicationPdf: MimeType = unsafe("application/pdf")

    extension (m: MimeType)
      def value: String = m
      def isImage: Boolean = m.startsWith("image/")
      def isVideo: Boolean = m.startsWith("video/")
      def isAudio: Boolean = m.startsWith("audio/")
      def isText: Boolean = m.startsWith("text/")

  // Base64Data: Base64-encoded data (must match base64 pattern)
  type Base64Constraint = Not[Empty] & Match["[A-Za-z0-9+/]+=*"]
  opaque type Base64Data = String :| Base64Constraint

  object Base64Data:
    def apply(value: String): Either[String, Base64Data] =
      value.refineEither[Base64Constraint]

    def unsafe(value: String): Base64Data =
      value.refineUnsafe[Base64Constraint]

    def fromBytes(bytes: Array[Byte]): Base64Data =
      unsafe(java.util.Base64.getEncoder.encodeToString(bytes))

    extension (b: Base64Data)
      def value: String = b
      def decode: Array[Byte] = java.util.Base64.getDecoder.decode(b)

  // Uri: URI/URL string
  type UriConstraint = Not[Empty]
  opaque type Uri = String :| UriConstraint

  object Uri:
    def apply(value: String): Either[String, Uri] =
      value.refineEither[UriConstraint]

    def unsafe(value: String): Uri =
      value.refineUnsafe[UriConstraint]

    extension (u: Uri)
      def value: String = u
      def isGcs: Boolean = u.startsWith("gs://")
      def isHttp: Boolean = u.startsWith("http://") || u.startsWith("https://")

  // Prompt: User prompt text (non-empty)
  opaque type Prompt = String :| Not[Empty]

  object Prompt:
    def apply(value: String): Either[String, Prompt] =
      value.refineEither[Not[Empty]]

    def unsafe(value: String): Prompt =
      value.refineUnsafe[Not[Empty]]

    extension (p: Prompt)
      def value: String = p
      def wordCount: Int = p.split("\\s+").length
      def charCount: Int = p.length
