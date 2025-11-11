package com.google.genai.scala.constraints

import io.github.iltotore.iron.*
import io.github.iltotore.iron.constraint.all.*

/**
 * Ultra-constrained numeric types using Iron refinement types.
 * All constraints are validated at compile-time when using literals,
 * or at runtime with descriptive errors when using dynamic values.
 */
object NumericConstraints:

  // Temperature: Controls randomness in generation (0.0 = deterministic, 2.0 = maximum randomness)
  type TemperatureConstraint = GreaterEqual[0.0] & LessEqual[2.0]
  opaque type Temperature = Double :| TemperatureConstraint

  object Temperature:
    def apply(value: Double): Either[String, Temperature] =
      value.refineEither[TemperatureConstraint]

    def unsafe(value: Double): Temperature =
      value.refineUnsafe[TemperatureConstraint]

    extension (t: Temperature)
      def value: Double = t

  // TopP: Nucleus sampling parameter (0.0 = deterministic, 1.0 = maximum diversity)
  type TopPConstraint = GreaterEqual[0.0] & LessEqual[1.0]
  opaque type TopP = Double :| TopPConstraint

  object TopP:
    def apply(value: Double): Either[String, TopP] =
      value.refineEither[TopPConstraint]

    def unsafe(value: Double): TopP =
      value.refineUnsafe[TopPConstraint]

    extension (t: TopP)
      def value: Double = t

  // TopK: Token selection count (must be positive)
  type TopKConstraint = Greater[0]
  opaque type TopK = Int :| TopKConstraint

  object TopK:
    def apply(value: Int): Either[String, TopK] =
      value.refineEither[TopKConstraint]

    def unsafe(value: Int): TopK =
      value.refineUnsafe[TopKConstraint]

    extension (t: TopK)
      def value: Int = t

  // MaxOutputTokens: Maximum response length
  type MaxOutputTokensConstraint = Positive
  opaque type MaxOutputTokens = Int :| MaxOutputTokensConstraint

  object MaxOutputTokens:
    def apply(value: Int): Either[String, MaxOutputTokens] =
      value.refineEither[MaxOutputTokensConstraint]

    def unsafe(value: Int): MaxOutputTokens =
      value.refineUnsafe[MaxOutputTokensConstraint]

    extension (m: MaxOutputTokens)
      def value: Int = m

  // CandidateCount: Number of response variations (1-8)
  type CandidateCountConstraint = GreaterEqual[1] & LessEqual[8]
  opaque type CandidateCount = Int :| CandidateCountConstraint

  object CandidateCount:
    def apply(value: Int): Either[String, CandidateCount] =
      value.refineEither[CandidateCountConstraint]

    def unsafe(value: Int): CandidateCount =
      value.refineUnsafe[CandidateCountConstraint]

    def default: CandidateCount = unsafe(1)

    extension (c: CandidateCount)
      def value: Int = c

  // AspectRatio: Image aspect ratio (width/height, must be positive)
  type AspectRatioConstraint = Positive
  opaque type AspectRatio = Double :| AspectRatioConstraint

  object AspectRatio:
    def apply(width: Int, height: Int): Either[String, AspectRatio] =
      if height == 0 then Left("Height cannot be zero")
      else (width.toDouble / height.toDouble).refineEither[AspectRatioConstraint]

    def unsafe(value: Double): AspectRatio =
      value.refineUnsafe[AspectRatioConstraint]

    // Common aspect ratios
    val Square: AspectRatio = unsafe(1.0)
    val Landscape_16_9: AspectRatio = unsafe(16.0 / 9.0)
    val Portrait_9_16: AspectRatio = unsafe(9.0 / 16.0)
    val Landscape_4_3: AspectRatio = unsafe(4.0 / 3.0)
    val Portrait_3_4: AspectRatio = unsafe(3.0 / 4.0)

    extension (a: AspectRatio)
      def value: Double = a

  // ThinkingBudget: Maximum tokens for model thinking (1024-8192)
  type ThinkingBudgetConstraint = GreaterEqual[1024] & LessEqual[8192]
  opaque type ThinkingBudget = Int :| ThinkingBudgetConstraint

  object ThinkingBudget:
    def apply(value: Int): Either[String, ThinkingBudget] =
      value.refineEither[ThinkingBudgetConstraint]

    def unsafe(value: Int): ThinkingBudget =
      value.refineUnsafe[ThinkingBudgetConstraint]

    extension (t: ThinkingBudget)
      def value: Int = t

  // FileSizeBytes: File size in bytes (max 2GB)
  type FileSizeBytesConstraint = Positive & LessEqual[2_147_483_648L]
  opaque type FileSizeBytes = Long :| FileSizeBytesConstraint

  object FileSizeBytes:
    def apply(value: Long): Either[String, FileSizeBytes] =
      value.refineEither[FileSizeBytesConstraint]

    def unsafe(value: Long): FileSizeBytes =
      value.refineUnsafe[FileSizeBytesConstraint]

    extension (f: FileSizeBytes)
      def value: Long = f
      def toMB: Double = f.toDouble / 1_048_576.0
      def toGB: Double = f.toDouble / 1_073_741_824.0

  // ImageDimension: Image width/height (must be positive, typically 64-4096)
  type ImageDimensionConstraint = GreaterEqual[64] & LessEqual[4096]
  opaque type ImageDimension = Int :| ImageDimensionConstraint

  object ImageDimension:
    def apply(value: Int): Either[String, ImageDimension] =
      value.refineEither[ImageDimensionConstraint]

    def unsafe(value: Int): ImageDimension =
      value.refineUnsafe[ImageDimensionConstraint]

    extension (d: ImageDimension)
      def value: Int = d
