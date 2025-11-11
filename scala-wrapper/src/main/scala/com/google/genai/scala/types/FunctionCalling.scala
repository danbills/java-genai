package com.google.genai.scala.types

import com.google.genai.scala.constraints.StringConstraints.*
import com.google.genai.scala.constraints.NumericConstraints.*
import io.github.iltotore.iron.*
import io.github.iltotore.iron.constraint.all.*
import cats.effect.*
import cats.syntax.all.*

/**
 * Ultra-constrained function calling types for agentic LLM behavior.
 *
 * This module provides:
 * - Type-safe function declarations
 * - Schema-validated parameters
 * - Automatic function execution
 * - Multi-turn conversation loops
 * - Error handling with Either
 */

// ============================================================================
// FUNCTION DECLARATION
// ============================================================================

/**
 * Function name with constraints.
 * Must be valid identifier: alphanumeric + underscores, start with letter.
 */
type FunctionNameConstraint = Not[Empty] & Match["[a-zA-Z][a-zA-Z0-9_]*"]
opaque type FunctionName = String :| FunctionNameConstraint

object FunctionName:
  def apply(value: String): Either[String, FunctionName] =
    value.refineEither[FunctionNameConstraint]

  def unsafe(value: String): FunctionName =
    value.refineUnsafe[FunctionNameConstraint]

  extension (f: FunctionName)
    def value: String = f

/**
 * Function description (required for LLM to understand when to call).
 * Should be clear, concise, and describe what the function does.
 */
type FunctionDescriptionConstraint = Not[Empty] & MinLength[10]
opaque type FunctionDescription = String :| FunctionDescriptionConstraint

object FunctionDescription:
  def apply(value: String): Either[String, FunctionDescription] =
    value.refineEither[FunctionDescriptionConstraint]

  def unsafe(value: String): FunctionDescription =
    value.refineUnsafe[FunctionDescriptionConstraint]

  extension (f: FunctionDescription)
    def value: String = f

/**
 * Parameter name (must be valid identifier).
 */
opaque type ParameterName = String :| FunctionNameConstraint

object ParameterName:
  def apply(value: String): Either[String, ParameterName] =
    value.refineEither[FunctionNameConstraint]

  def unsafe(value: String): ParameterName =
    value.refineUnsafe[FunctionNameConstraint]

  extension (p: ParameterName)
    def value: String = p

/**
 * Parameter description (guides LLM on what to provide).
 */
opaque type ParameterDescription = String :| Not[Empty]

object ParameterDescription:
  def apply(value: String): Either[String, ParameterDescription] =
    value.refineEither[Not[Empty]]

  def unsafe(value: String): ParameterDescription =
    value.refineUnsafe[Not[Empty]]

  extension (p: ParameterDescription)
    def value: String = p

// ============================================================================
// FUNCTION PARAMETERS
// ============================================================================

/**
 * Function parameter with schema and constraints.
 */
case class FunctionParameter(
  name: ParameterName,
  description: ParameterDescription,
  schema: Schema,
  required: Boolean = true
) derives CanEqual

object FunctionParameter:
  /** Create a required string parameter */
  def string(name: String, description: String): FunctionParameter =
    FunctionParameter(
      ParameterName.unsafe(name),
      ParameterDescription.unsafe(description),
      Schema.string(description),
      required = true
    )

  /** Create an optional string parameter */
  def optionalString(name: String, description: String): FunctionParameter =
    FunctionParameter(
      ParameterName.unsafe(name),
      ParameterDescription.unsafe(description),
      Schema.string(description),
      required = false
    )

  /** Create a required integer parameter */
  def integer(name: String, description: String, min: Option[Int] = None, max: Option[Int] = None): FunctionParameter =
    FunctionParameter(
      ParameterName.unsafe(name),
      ParameterDescription.unsafe(description),
      Schema.integer(description, min, max),
      required = true
    )

  /** Create a required boolean parameter */
  def boolean(name: String, description: String): FunctionParameter =
    FunctionParameter(
      ParameterName.unsafe(name),
      ParameterDescription.unsafe(description),
      Schema.boolean(description),
      required = true
    )

  /** Create a required array parameter */
  def array(name: String, description: String, itemSchema: Schema): FunctionParameter =
    FunctionParameter(
      ParameterName.unsafe(name),
      ParameterDescription.unsafe(description),
      Schema.array(itemSchema, description),
      required = true
    )

  /** Create an enum parameter */
  def enum(name: String, description: String, values: List[String]): FunctionParameter =
    FunctionParameter(
      ParameterName.unsafe(name),
      ParameterDescription.unsafe(description),
      Schema(SchemaType.String, Some(NonEmptyString.unsafe(description)), enum = values),
      required = true
    )

// ============================================================================
// FUNCTION DECLARATION
// ============================================================================

/**
 * Complete function declaration with parameters.
 */
case class FunctionDeclaration(
  name: FunctionName,
  description: FunctionDescription,
  parameters: List[FunctionParameter]
) derives CanEqual:

  /** Convert to Schema for API */
  def toSchema: Schema =
    val properties = parameters.map(p => p.name.value -> p.schema).toMap
    val required = parameters.filter(_.required).map(p => NonEmptyString.unsafe(p.name.value))
    Schema.obj(
      properties = properties,
      required = required,
      description = description.value
    )

  /** Get parameter by name */
  def getParameter(name: String): Option[FunctionParameter] =
    parameters.find(_.name.value == name)

object FunctionDeclaration:
  /** Builder for function declarations */
  def apply(
    name: String,
    description: String,
    parameters: FunctionParameter*
  ): FunctionDeclaration =
    new FunctionDeclaration(
      FunctionName.unsafe(name),
      FunctionDescription.unsafe(description),
      parameters.toList
    )

// ============================================================================
// FUNCTION EXECUTION
// ============================================================================

/**
 * Type-safe function executor.
 * Maps function names to implementations.
 */
trait FunctionExecutor[F[_]]:
  /** Execute a function with given arguments */
  def execute(name: FunctionName, args: Map[String, Any]): F[FunctionResult]

  /** Check if function is available */
  def hasFunction(name: FunctionName): Boolean

  /** Get all available function declarations */
  def declarations: List[FunctionDeclaration]

/**
 * Result of function execution.
 */
sealed trait FunctionResult derives CanEqual

object FunctionResult:
  /** Successful execution with result */
  case class Success(value: Map[String, Any]) extends FunctionResult

  /** Successful execution with simple string result */
  case class TextSuccess(message: NonEmptyString) extends FunctionResult

  /** Function execution failed */
  case class Error(message: NonEmptyString) extends FunctionResult

  /** Convenient constructors */
  def success(value: Map[String, Any]): FunctionResult = Success(value)
  def success(message: String): FunctionResult = TextSuccess(NonEmptyString.unsafe(message))
  def error(message: String): FunctionResult = Error(NonEmptyString.unsafe(message))

// ============================================================================
// FUNCTION EXECUTION CONTEXT
// ============================================================================

/**
 * Execution context for a function call.
 * Contains metadata about the call.
 */
case class FunctionCallContext(
  conversationId: Option[NonEmptyString] = None,
  userId: Option[NonEmptyString] = None,
  timestamp: java.time.Instant = java.time.Instant.now,
  metadata: Map[String, String] = Map.empty
) derives CanEqual

/**
 * Function call request from LLM.
 */
case class FunctionCallRequest(
  name: FunctionName,
  arguments: Map[String, Any],
  callId: Option[NonEmptyString] = None
) derives CanEqual

/**
 * Function call response to LLM.
 */
case class FunctionCallResponse(
  name: FunctionName,
  result: FunctionResult,
  callId: Option[NonEmptyString] = None
) derives CanEqual:

  /** Convert to Part for inclusion in Content */
  def toPart: Part =
    val response = result match
      case FunctionResult.Success(value) => value
      case FunctionResult.TextSuccess(msg) => Map("result" -> msg.value)
      case FunctionResult.Error(msg) => Map("error" -> msg.value)

    Part.FunctionResponse(NonEmptyString.unsafe(name.value), response)

// ============================================================================
// AUTOMATIC FUNCTION CALLING (AFC) CONFIG
// ============================================================================

/**
 * Maximum iterations for automatic function calling loop.
 * Prevents infinite loops.
 */
type MaxIterationsConstraint = GreaterEqual[1] & LessEqual[50]
opaque type MaxIterations = Int :| MaxIterationsConstraint

object MaxIterations:
  def apply(value: Int): Either[String, MaxIterations] =
    value.refineEither[MaxIterationsConstraint]

  def unsafe(value: Int): MaxIterations =
    value.refineUnsafe[MaxIterationsConstraint]

  def default: MaxIterations = unsafe(10)

  extension (m: MaxIterations)
    def value: Int = m

/**
 * Configuration for automatic function calling.
 */
case class AutomaticFunctionCallingConfig(
  maxIterations: MaxIterations = MaxIterations.default,
  functionCallingMode: FunctionCallingMode = FunctionCallingMode.Auto,
  throwOnError: Boolean = false,
  verbose: Boolean = false
) derives CanEqual

object AutomaticFunctionCallingConfig:
  /** Conservative: fewer iterations, stricter error handling */
  def conservative: AutomaticFunctionCallingConfig =
    AutomaticFunctionCallingConfig(
      maxIterations = MaxIterations.unsafe(5),
      functionCallingMode = FunctionCallingMode.Auto,
      throwOnError = true,
      verbose = false
    )

  /** Aggressive: more iterations, lenient error handling */
  def aggressive: AutomaticFunctionCallingConfig =
    AutomaticFunctionCallingConfig(
      maxIterations = MaxIterations.unsafe(20),
      functionCallingMode = FunctionCallingMode.Auto,
      throwOnError = false,
      verbose = true
    )

// ============================================================================
// AGENTIC CONVERSATION STATE
// ============================================================================

/**
 * State of an agentic conversation.
 * Tracks history and function calls.
 */
case class AgenticConversation(
  history: List[Content],
  functionCalls: List[FunctionCallRequest] = Nil,
  functionResponses: List[FunctionCallResponse] = Nil,
  iterationCount: Int = 0
) derives CanEqual:

  /** Add user message */
  def addUserMessage(message: Prompt): AgenticConversation =
    copy(history = history :+ Content(Role.User, List(Part.Text(message))))

  /** Add model response */
  def addModelResponse(content: Content): AgenticConversation =
    copy(history = history :+ content)

  /** Add function call */
  def addFunctionCall(call: FunctionCallRequest): AgenticConversation =
    copy(functionCalls = functionCalls :+ call)

  /** Add function response */
  def addFunctionResponse(response: FunctionCallResponse): AgenticConversation =
    val responseContent = Content(
      Role.User,
      List(response.toPart)
    )
    copy(
      functionResponses = functionResponses :+ response,
      history = history :+ responseContent
    )

  /** Increment iteration count */
  def incrementIteration: AgenticConversation =
    copy(iterationCount = iterationCount + 1)

  /** Check if max iterations reached */
  def hasReachedMaxIterations(max: MaxIterations): Boolean =
    iterationCount >= max.value

  /** Get the last message text */
  def lastMessageText: Option[String] =
    history.lastOption.flatMap { content =>
      content.parts.collectFirst {
        case Part.Text(text) => text.value
      }
    }

object AgenticConversation:
  /** Start a new conversation */
  def empty: AgenticConversation =
    AgenticConversation(history = Nil)

  /** Start with a system prompt */
  def withSystemPrompt(prompt: Prompt): AgenticConversation =
    AgenticConversation(
      history = List(Content(Role.System, List(Part.Text(prompt))))
    )

// ============================================================================
// TOOL BUILDERS
// ============================================================================

/**
 * Builder for creating Tool.FunctionDeclaration from FunctionDeclaration.
 */
extension (fd: FunctionDeclaration)
  def toTool: Tool =
    Tool.FunctionDeclaration(
      name = NonEmptyString.unsafe(fd.name.value),
      description = NonEmptyString.unsafe(fd.description.value),
      parameters = fd.toSchema
    )

/**
 * Convenient builders for tools.
 */
object Tools:
  /** Code execution tool */
  def codeExecution: Tool = Tool.CodeExecution

  /** Google Search tool */
  def googleSearch: Tool = Tool.GoogleSearch

  /** Function declaration tool */
  def function(declaration: FunctionDeclaration): Tool = declaration.toTool

  /** Multiple function declarations */
  def functions(declarations: FunctionDeclaration*): List[Tool] =
    declarations.map(_.toTool).toList

// ============================================================================
// HELPER TYPES
// ============================================================================

/**
 * Extract function calls from model response.
 */
extension (response: GenerateContentResponse)
  def functionCalls: List[FunctionCallRequest] =
    response.candidates.flatMap { candidate =>
      candidate.content.parts.collect {
        case Part.FunctionCall(name, args) =>
          FunctionCallRequest(
            name = FunctionName.unsafe(name.value),
            arguments = args
          )
      }
    }

  def hasFunctionCalls: Boolean =
    functionCalls.nonEmpty

  def isComplete: Boolean =
    response.candidates.exists { candidate =>
      candidate.finishReason.contains(FinishReason.Stop)
    }
