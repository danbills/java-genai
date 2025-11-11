# Design Philosophy: Ultra-Constrained Scala Wrapper

## Core Principles

### 1. **Make Illegal States Unrepresentable**

The type system prevents impossible combinations:

```scala
// ❌ Can't compile: Gemini-only operation on Vertex client
vertexClient.uploadFile(...)

// ❌ Can't compile: Wrong model for operation
client.generateImages(gemini20Flash, ...)  // Text model, not image

// ❌ Can't compile: Invalid temperature range
Temperature.unsafe(5.0)  // Will throw at runtime with clear message
```

### 2. **Zero-Cost Abstractions**

All type safety is compile-time only. Runtime has zero overhead:

```scala
opaque type Temperature = Double :| TemperatureConstraint
// At runtime: just a Double
// At compile-time: fully type-checked with constraints
```

### 3. **Fail Fast with Clear Errors**

All validation happens eagerly with descriptive errors:

```scala
Temperature(3.0) // Either[String, Temperature] =
                 // Left("Should be less than or equal to 2.0")

ProjectId("INVALID") // Either[String, ProjectId] =
                     // Left("Should match [a-z][a-z0-9-]{4,28}[a-z0-9]")
```

### 4. **Phantom Types for API Variants**

Prevents mixing incompatible APIs at compile-time:

```scala
trait GeminiApiClient extends GenAiClient[GeminiApi]
trait VertexApiClient extends GenAiClient[VertexApi]

// Type parameter tracks which API you're using
// Compiler ensures you can't mix them
```

### 5. **Capability-Based Model Types**

Models carry their capabilities in their types:

```scala
Model[ApiVariant, Capabilities]

gemini20Flash: Model[A, TextGeneration & FunctionCalling & CodeExecution]
imagen3: Model[A, ImageGeneration]

// Operations require specific capabilities via type constraints
```

## Type System Architecture

### Constraint Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    USER-FACING API                          │
│  GenAiClient[A <: ApiVariant]                               │
│  - Type-safe operations                                     │
│  - Capability constraints                                   │
│  - Feature availability                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              PHANTOM TYPE LAYER                             │
│  - ApiVariant: GeminiApi | VertexApi                        │
│  - ModelCapability: TextGeneration | ImageGeneration | ...  │
│  - Feature availability proofs                              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              DOMAIN MODEL LAYER                             │
│  - ADTs for all domain concepts (sealed traits)             │
│  - Case classes for structured data                         │
│  - Exhaustive pattern matching                              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           IRON REFINEMENT LAYER                             │
│  - Opaque types with constraints                            │
│  - Compile-time literal validation                          │
│  - Runtime validation with Either                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              PRIMITIVE LAYER                                │
│  - String, Double, Int, Long                                │
│  - Zero runtime overhead                                    │
└─────────────────────────────────────────────────────────────┘
```

## Iron Types Deep Dive

### Why Iron?

Iron provides **refinement types** - types that carry compile-time or runtime constraints:

```scala
// Basic refinement
type Positive = Greater[0]
val x: Int :| Positive = 42  // OK
val y: Int :| Positive = -5  // Compile error (if literal) or runtime error

// Compound refinements
type TemperatureConstraint = GreaterEqual[0.0] & LessEqual[2.0]
type Temperature = Double :| TemperatureConstraint
```

### Opaque Types + Iron = Perfect Match

```scala
opaque type Temperature = Double :| TemperatureConstraint

object Temperature:
  // Safe constructor returns Either
  def apply(value: Double): Either[String, Temperature] =
    value.refineEither[TemperatureConstraint]

  // Unsafe constructor for literals (validated at compile-time when possible)
  def unsafe(value: Double): Temperature =
    value.refineUnsafe[TemperatureConstraint]

  // Extension methods
  extension (t: Temperature)
    def value: Double = t
```

**Benefits:**
- Type is opaque: can't accidentally use a `Double` where `Temperature` expected
- Constraint is enforced: can't create invalid `Temperature`
- Zero runtime cost: erased to `Double`
- Clear errors: `Left("Should be less than or equal to 2.0")`

## Phantom Type Pattern

### API Variant Separation

```scala
sealed trait ApiVariant
sealed trait GeminiApi extends ApiVariant
sealed trait VertexApi extends ApiVariant

// Client parameterized by variant
trait GenAiClient[A <: ApiVariant]:
  def generateContent[Caps](...)
    (using HasCapability[Caps, TextGeneration]): IO[Response]

// Gemini-specific operations
trait GeminiApiClient extends GenAiClient[GeminiApi]:
  def uploadFile(...): IO[FileId]

// Vertex-specific operations
trait VertexApiClient extends GenAiClient[VertexApi]:
  def createTuningJob(...): IO[TuningJobId]
```

**How it works:**
- `A <: ApiVariant` is a phantom type (not used at runtime)
- Methods return types parameterized by `A`
- Compiler tracks which API variant you're using
- Prevents calling wrong operations on wrong client

### Model Capabilities

```scala
sealed trait ModelCapability
sealed trait TextGeneration extends ModelCapability
sealed trait ImageGeneration extends ModelCapability

// Model carries capabilities in type
case class Model[A <: ApiVariant, Caps <: ModelCapability](id: ModelId)

// Operations require capabilities
def generateImages[Caps](model: Model[A, Caps], ...)
  (using HasCapability[Caps, ImageGeneration]): IO[List[Image]]
```

**How it works:**
- `Caps` is a phantom type encoding capabilities
- Type intersections: `TextGeneration & FunctionCalling`
- `HasCapability` type class proves capability exists
- Compiler checks capability at call site

## Type Class Pattern

### HasCapability Type Class

```scala
trait HasCapability[M <: ModelCapability, Cap <: ModelCapability]

object HasCapability:
  // Base case: capability has itself
  given reflexive[C <: ModelCapability]: HasCapability[C, C] =
    new HasCapability {}

  // Intersection cases
  given intersectionLeft[C1, C2]: HasCapability[C1 & C2, C1] =
    new HasCapability {}

  given intersectionRight[C1, C2]: HasCapability[C1 & C2, C2] =
    new HasCapability {}
```

**How it works:**
- `using HasCapability[Caps, TextGeneration]` requires proof
- Compiler searches for given instances
- If no instance found, compilation fails
- Proof is erased at runtime (zero cost)

### FeatureAvailable Type Class

```scala
trait FeatureAvailable[F <: Feature[A], A <: ApiVariant]

object FeatureAvailable:
  // Files only in Gemini
  given filesInGemini: FeatureAvailable[Feature.Files.type, GeminiApi] =
    new FeatureAvailable {}

  // Tuning only in Vertex
  given tuningInVertex: FeatureAvailable[Feature.Tuning.type, VertexApi] =
    new FeatureAvailable {}
```

## ADT Pattern

### Why Sealed Traits?

```scala
sealed trait Role
object Role:
  case object User extends Role
  case object Model extends Role
  case object System extends Role

// Pattern matching is exhaustive
def describe(role: Role): String = role match
  case Role.User => "user"
  case Role.Model => "model"
  case Role.System => "system"
  // Compiler ensures all cases handled!
```

**Benefits:**
- Closed set of values
- Exhaustive pattern matching
- Can't add cases outside file
- Compiler catches missing cases

### Complex ADTs

```scala
sealed trait Part
object Part:
  case class Text(text: Prompt) extends Part
  case class InlineData(data: Base64Data, mimeType: MimeType) extends Part
  case class FileData(uri: Uri, mimeType: MimeType) extends Part
  // ... more cases

// All fields are iron-constrained types
// Pattern matching is exhaustive
// Impossible to create invalid Part
```

## Comparison with Java SDK

### Java SDK (Stringly-Typed)

```java
// ❌ No type safety
GenerationConfig config = GenerationConfig.builder()
  .temperature(5.0)  // Invalid! But compiles
  .topP(1.5)         // Invalid! But compiles
  .build();

// ❌ Can mix APIs
GeminiClient client = new GeminiClient(apiKey);
client.createTuningJob(...);  // Runtime error!

// ❌ No capability checking
Model model = Model.get("imagen-3.0");
client.generateContent(model, "text");  // Runtime error!

// ❌ Nulls everywhere
String text = response.getText();  // Might be null!
```

### Scala Wrapper (Ultra-Constrained)

```scala
// ✅ Type-safe configuration
val config = for
  temp <- Temperature(5.0)  // Returns Left("Should be <= 2.0")
  topP <- TopP(1.5)         // Returns Left("Should be <= 1.0")
yield GenerationConfig(temperature = Some(temp), topP = Some(topP))
// config: Either[String, GenerationConfig] = Left(...)

// ✅ Can't mix APIs
val client: GeminiApiClient = ...
// client.createTuningJob(...)  // Won't compile!

// ✅ Capability checking
val model: Model[GeminiApi, ImageGeneration] = Model.imagen3
// client.generateContent(model, ...)  // Won't compile!

// ✅ No nulls
val text: Option[String] = response.firstText  // Explicit Option
```

## Performance Characteristics

All type safety is **compile-time only**:

```scala
// This code:
val temp: Temperature = Temperature.unsafe(1.0)
val config = GenerationConfig(temperature = Some(temp))

// Compiles to (equivalent of):
val config = GenerationConfig(temperature = Some(1.0))

// Zero overhead!
```

**Runtime costs:**
- Opaque types: **0 bytes** (erased to primitives)
- Phantom types: **0 bytes** (erased completely)
- Type class instances: **0 bytes** (erased)
- ADTs: **same as case classes** (JVM objects)

## Error Messages

### Compile-Time Errors

```scala
client.generateImages(gemini20Flash, ...)
```

**Error:**
```
No given instance of HasCapability[
  TextGeneration & FunctionCalling & CodeExecution,
  ImageGeneration
] was found
```

**Translation:** Model doesn't have ImageGeneration capability.

### Runtime Errors

```scala
Temperature(3.0)
```

**Error:**
```
Left("Should be less than or equal to 2.0")
```

**Translation:** Temperature must be in range [0.0, 2.0].

## Extension Points

### Adding New Constraints

```scala
// Add new iron constraint
type PositiveEven = Positive & Multiple[2]
opaque type EvenNumber = Int :| PositiveEven

object EvenNumber:
  def apply(value: Int): Either[String, EvenNumber] =
    value.refineEither[PositiveEven]
```

### Adding New Capabilities

```scala
// Add new model capability
sealed trait AudioGeneration extends ModelCapability

// Add to model type
type AudioModel = Model[GeminiApi, AudioGeneration]

// Add operation
def generateAudio[Caps](model: Model[A, Caps], ...)
  (using HasCapability[Caps, AudioGeneration]): IO[Audio]
```

### Adding New API Variants

```scala
// Add new API variant
sealed trait LocalApi extends ApiVariant

// Add new client
trait LocalApiClient extends GenAiClient[LocalApi]:
  def runLocally(...): IO[Response]
```

## Lessons Learned

### What Works Well

1. **Iron for validation** - Best refinement type library for Scala 3
2. **Phantom types for variants** - Prevents API mixing with zero cost
3. **Capability types** - Makes impossible operations unrepresentable
4. **ADTs for domain** - Exhaustive matching catches errors
5. **Opaque types** - Zero-cost type safety

### Tradeoffs

1. **Learning curve** - Advanced type features require understanding
2. **Compile times** - Heavy type-level computation can slow compilation
3. **Error messages** - Can be cryptic for type class resolution
4. **Java interop** - Requires bridge code to underlying SDK

### When to Use This Pattern

**Use when:**
- Safety is critical (medical, financial, security)
- Domain has clear constraints (ranges, formats, enums)
- Multiple incompatible modes (APIs, backends, environments)
- Long-lived codebase (refactoring safety matters)

**Don't use when:**
- Prototype/MVP (overhead not worth it)
- Simple domain (few constraints)
- Performance critical (compilation time matters)
- Team unfamiliar with advanced types

## Future Enhancements

1. **Scala 3 macros** for automatic ADT derivation
2. **Literal types** for model IDs (`"gemini-2.0-flash"` as a type)
3. **Match types** for capability intersection simplification
4. **Metaprogramming** for schema derivation from case classes
5. **ZIO/Cats Effect integration** for streaming responses

## References

- [Iron Library](https://github.com/Iltotore/iron)
- [Opaque Types](https://docs.scala-lang.org/scala3/reference/other-new-features/opaques.html)
- [Phantom Types](https://blog.rockthejvm.com/phantom-types/)
- [Type Classes in Scala 3](https://docs.scala-lang.org/scala3/book/ca-type-classes.html)
- [ADTs in Scala](https://docs.scala-lang.org/scala3/book/types-adts-gadts.html)
