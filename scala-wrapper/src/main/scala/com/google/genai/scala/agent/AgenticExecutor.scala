package com.google.genai.scala.agent

import cats.effect.*
import cats.syntax.all.*
import com.google.genai.scala.client.*
import com.google.genai.scala.types.*
import com.google.genai.scala.types.PhantomTypes.*
import com.google.genai.scala.constraints.StringConstraints.*

/**
 * Agentic executor for automatic function calling loops.
 *
 * This implements the core agent loop:
 * 1. User sends message to LLM
 * 2. LLM responds (possibly with function calls)
 * 3. Functions are executed automatically
 * 4. Results are sent back to LLM
 * 5. Loop continues until LLM stops calling functions
 *
 * Uses iron types to ensure:
 * - Max iterations is bounded (1-50)
 * - Function names are valid
 * - All parameters are validated
 */
trait AgenticExecutor[F[_], A <: ApiVariant]:

  /**
   * Execute a single turn: send prompt, handle function calls, return final response.
   */
  def executeTurn(
    conversation: AgenticConversation,
    model: Model[A, _ <: ModelCapability],
    config: GenerationConfig = GenerationConfig(),
    afcConfig: AutomaticFunctionCallingConfig = AutomaticFunctionCallingConfig()
  ): F[(AgenticConversation, GenerateContentResponse)]

  /**
   * Execute full agentic loop with automatic function calling.
   * Returns when LLM stops calling functions or max iterations reached.
   */
  def executeLoop(
    initialPrompt: Prompt,
    model: Model[A, _ <: ModelCapability],
    config: GenerationConfig = GenerationConfig(),
    afcConfig: AutomaticFunctionCallingConfig = AutomaticFunctionCallingConfig()
  ): F[AgenticResult]

  /**
   * Chat interface: maintain conversation across multiple user inputs.
   */
  def chat(
    conversation: AgenticConversation,
    userMessage: Prompt,
    model: Model[A, _ <: ModelCapability],
    config: GenerationConfig = GenerationConfig(),
    afcConfig: AutomaticFunctionCallingConfig = AutomaticFunctionCallingConfig()
  ): F[AgenticResult]

/**
 * Result of agentic execution.
 */
case class AgenticResult(
  conversation: AgenticConversation,
  finalResponse: GenerateContentResponse,
  functionCallCount: Int,
  iterationCount: Int,
  completed: Boolean,
  error: Option[String] = None
) derives CanEqual:

  /** Get final text response */
  def finalText: Option[String] = finalResponse.firstText

  /** Get all function calls made */
  def allFunctionCalls: List[FunctionCallRequest] = conversation.functionCalls

  /** Get all function responses */
  def allFunctionResponses: List[FunctionCallResponse] = conversation.functionResponses

  /** Check if max iterations was reached */
  def reachedMaxIterations: Boolean = !completed && error.isEmpty

/**
 * Implementation of agentic executor.
 */
class AgenticExecutorImpl[A <: ApiVariant](
  client: GenAiClient[A],
  executor: FunctionExecutor[IO]
) extends AgenticExecutor[IO, A]:

  def executeTurn(
    conversation: AgenticConversation,
    model: Model[A, _ <: ModelCapability],
    config: GenerationConfig,
    afcConfig: AutomaticFunctionCallingConfig
  ): IO[(AgenticConversation, GenerateContentResponse)] =
    for
      // Generate content with current history
      response <- client.generateContentWithHistory(
        model = model,
        history = conversation.history,
        config = config,
        safetySettings = Nil
      )

      // Update conversation with model response
      modelContent = response.candidates.headOption
        .map(_.content)
        .getOrElse(Content(Role.Model, Nil))

      updatedConversation = conversation.addModelResponse(modelContent)

    yield (updatedConversation, response)

  def executeLoop(
    initialPrompt: Prompt,
    model: Model[A, _ <: ModelCapability],
    config: GenerationConfig,
    afcConfig: AutomaticFunctionCallingConfig
  ): IO[AgenticResult] =
    val initialConversation = AgenticConversation.empty.addUserMessage(initialPrompt)
    executeLoopRecursive(initialConversation, model, config, afcConfig)

  def chat(
    conversation: AgenticConversation,
    userMessage: Prompt,
    model: Model[A, _ <: ModelCapability],
    config: GenerationConfig,
    afcConfig: AutomaticFunctionCallingConfig
  ): IO[AgenticResult] =
    val updatedConversation = conversation.addUserMessage(userMessage)
    executeLoopRecursive(updatedConversation, model, config, afcConfig)

  /**
   * Recursive function calling loop.
   */
  private def executeLoopRecursive(
    conversation: AgenticConversation,
    model: Model[A, _ <: ModelCapability],
    config: GenerationConfig,
    afcConfig: AutomaticFunctionCallingConfig,
    functionCallCount: Int = 0
  ): IO[AgenticResult] =
    // Check max iterations
    if conversation.hasReachedMaxIterations(afcConfig.maxIterations) then
      IO.pure(
        AgenticResult(
          conversation = conversation,
          finalResponse = GenerateContentResponse(Nil, None, None),
          functionCallCount = functionCallCount,
          iterationCount = conversation.iterationCount,
          completed = false,
          error = Some(s"Max iterations (${afcConfig.maxIterations.value}) reached")
        )
      )
    else
      for
        // Log iteration if verbose
        _ <- IO.whenA(afcConfig.verbose)(
          IO.println(s"[Agent] Iteration ${conversation.iterationCount + 1}/${afcConfig.maxIterations.value}")
        )

        // Execute one turn
        (updatedConv, response) <- executeTurn(
          conversation.incrementIteration,
          model,
          config,
          afcConfig
        )

        // Check for function calls
        functionCalls = response.functionCalls

        result <- if functionCalls.isEmpty then
          // No function calls - we're done!
          IO.whenA(afcConfig.verbose)(
            IO.println(s"[Agent] Completed with ${functionCallCount} function calls")
          ) *>
          IO.pure(
            AgenticResult(
              conversation = updatedConv,
              finalResponse = response,
              functionCallCount = functionCallCount,
              iterationCount = updatedConv.iterationCount,
              completed = true
            )
          )
        else
          // Execute function calls
          for
            _ <- IO.whenA(afcConfig.verbose)(
              IO.println(s"[Agent] Executing ${functionCalls.length} function calls")
            )

            // Execute all function calls
            responses <- functionCalls.traverse { call =>
              executeFunctionCall(call, afcConfig).map { result =>
                FunctionCallResponse(call.name, result, call.callId)
              }
            }

            // Add function responses to conversation
            convWithResponses = responses.foldLeft(updatedConv) { (conv, resp) =>
              conv.addFunctionCall(
                FunctionCallRequest(resp.name, Map.empty)
              ).addFunctionResponse(resp)
            }

            // Continue loop
            result <- executeLoopRecursive(
              convWithResponses,
              model,
              config,
              afcConfig,
              functionCallCount + functionCalls.length
            )

          yield result

      yield result

  /**
   * Execute a single function call.
   */
  private def executeFunctionCall(
    call: FunctionCallRequest,
    afcConfig: AutomaticFunctionCallingConfig
  ): IO[FunctionResult] =
    for
      _ <- IO.whenA(afcConfig.verbose)(
        IO.println(s"[Agent] Calling function: ${call.name.value}")
      )

      result <- executor.execute(call.name, call.arguments).handleErrorWith { err =>
        val errorMsg = s"Function execution failed: ${err.getMessage}"
        IO.whenA(afcConfig.verbose)(IO.println(s"[Agent] Error: $errorMsg")) *>
        (if afcConfig.throwOnError then IO.raiseError(err)
         else IO.pure(FunctionResult.error(errorMsg)))
      }

      _ <- IO.whenA(afcConfig.verbose)(
        result match
          case FunctionResult.Success(_) => IO.println(s"[Agent] Function succeeded")
          case FunctionResult.TextSuccess(msg) => IO.println(s"[Agent] Function returned: ${msg.value.take(50)}...")
          case FunctionResult.Error(msg) => IO.println(s"[Agent] Function error: ${msg.value}")
      )

    yield result

/**
 * Factory for creating agentic executors.
 */
object AgenticExecutor:

  /**
   * Create an agentic executor with a function executor.
   */
  def apply[A <: ApiVariant](
    client: GenAiClient[A],
    executor: FunctionExecutor[IO]
  ): AgenticExecutor[IO, A] =
    new AgenticExecutorImpl[A](client, executor)

  /**
   * Create an agentic executor with a map of functions.
   */
  def withFunctions[A <: ApiVariant](
    client: GenAiClient[A],
    functions: Map[FunctionName, (Map[String, Any]) => IO[FunctionResult]],
    declarations: List[FunctionDeclaration]
  ): AgenticExecutor[IO, A] =
    val executor = new FunctionExecutor[IO]:
      def execute(name: FunctionName, args: Map[String, Any]): IO[FunctionResult] =
        functions.get(name) match
          case Some(fn) => fn(args)
          case None => IO.pure(FunctionResult.error(s"Unknown function: ${name.value}"))

      def hasFunction(name: FunctionName): Boolean =
        functions.contains(name)

      def declarations: List[FunctionDeclaration] =
        declarations

    new AgenticExecutorImpl[A](client, executor)

  /**
   * Create an agentic executor with a simple function map.
   * Automatically creates function declarations.
   */
  def simple[A <: ApiVariant](
    client: GenAiClient[A],
    functions: (FunctionDeclaration, (Map[String, Any]) => IO[FunctionResult])*
  ): AgenticExecutor[IO, A] =
    val functionMap = functions.map { case (decl, fn) =>
      FunctionName.unsafe(decl.name.value) -> fn
    }.toMap

    val declarations = functions.map(_._1).toList

    withFunctions(client, functionMap, declarations)

/**
 * DSL for building function executors.
 */
object FunctionExecutorDSL:

  /**
   * Builder for creating function executors.
   */
  class FunctionExecutorBuilder:
    private var functions: Map[FunctionName, (Map[String, Any]) => IO[FunctionResult]] = Map.empty
    private var declarations: Map[FunctionName, FunctionDeclaration] = Map.empty

    /**
     * Register a function.
     */
    def register(
      declaration: FunctionDeclaration,
      implementation: Map[String, Any] => IO[FunctionResult]
    ): FunctionExecutorBuilder =
      val name = FunctionName.unsafe(declaration.name.value)
      functions = functions + (name -> implementation)
      declarations = declarations + (name -> declaration)
      this

    /**
     * Register a function with simple string result.
     */
    def registerSimple(
      declaration: FunctionDeclaration,
      implementation: Map[String, Any] => IO[String]
    ): FunctionExecutorBuilder =
      register(declaration, args => implementation(args).map(FunctionResult.success))

    /**
     * Build the executor.
     */
    def build: FunctionExecutor[IO] = new FunctionExecutor[IO]:
      def execute(name: FunctionName, args: Map[String, Any]): IO[FunctionResult] =
        functions.get(name) match
          case Some(fn) => fn(args)
          case None => IO.pure(FunctionResult.error(s"Unknown function: ${name.value}"))

      def hasFunction(name: FunctionName): Boolean =
        functions.contains(name)

      def declarations: List[FunctionDeclaration] =
        FunctionExecutorBuilder.this.declarations.values.toList

  /**
   * Start building a function executor.
   */
  def builder: FunctionExecutorBuilder = new FunctionExecutorBuilder
