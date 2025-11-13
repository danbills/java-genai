//> using scala "3.3.1"
//> using dep "org.typelevel::cats-effect::3.5.4"
//> using dep "co.fs2::fs2-core::3.10.2"
//> using dep "io.circe::circe-core::0.14.7"
//> using dep "io.circe::circe-generic::0.14.7"
//> using dep "io.circe::circe-parser::0.14.7"

// This is a HYPOTHETICAL dependency. It does not exist.
// It represents the scala-wrapper we have been designing.
//> using dep "com.google.genai::genai-scala-iron::0.1.0-SNAPSHOT"

import cats.effect.{IO, IOApp, ExitCode}
import cats.syntax.all.*
import scala.sys.process.*
import scala.util.Try

// --- Hypothetical imports from the scala-wrapper ---
// These types are based on the README.md and are what a real wrapper would provide.
import com.google.genai.scala.client.GenAiClient
import com.google.genai.scala.types.{Content, Part, Role, GenerationConfig, Tool => GenAiTool}
import com.google.genai.scala.types.Model.gemini20Flash // Assuming a model with function calling

// --- Strong, Semantic Types (from 01_ToolDefinition.scala) ---
object Types:
  opaque type ToolName = String
  object ToolName:
    def apply(name: String): ToolName = name
  extension (name: ToolName) def value: String = name

  opaque type ToolOutput = String
  object ToolOutput:
    def apply(output: String): ToolOutput = output
  extension (output: ToolOutput) def value: String = output
import Types.*

// --- Tool Definition (Adapted for the real wrapper) ---
trait Tool:
  def definition: GenAiTool // The tool definition for the GenAI API
  def execute(args: Map[String, String]): IO[ToolOutput]

object GitTool extends Tool:
  import com.google.genai.scala.types.Schema

  override val definition: GenAiTool = GenAiTool(
    name = "git",
    description = "Executes a git command locally. The input is a single string containing all arguments for the git command (e.g., 'status -s' or 'log --oneline -n 5').",
    schema = Schema.obj(
      properties = Map(
        "args" -> Schema.string("The arguments to pass to the git command.")
      ),
      required = List("args")
    )
  )

  override def execute(args: Map[String, String]): IO[ToolOutput] = IO.blocking {
    args.get("args") match
      case Some(rawArgs) =>
        // SECURITY WARNING: Executing arbitrary shell commands is dangerous.
        val command = Seq("git") ++ rawArgs.split(' ').filter(_.nonEmpty)
        val stdout = new StringBuilder
        val stderr = new StringBuilder
        val logger = ProcessLogger(stdout.append(_).append("\n"), stderr.append(_).append("\n"))
        command.!(logger)
        ToolOutput(s"STDOUT:\n${stdout.toString}\nSTDERR:\n${stderr.toString}")
      case None => ToolOutput("Error: Missing required 'args' parameter for the git tool.")
  }

// --- Agent Components ---
object ToolDispatcher:
  private val tools: Map[String, Tool] = List(GitTool).map(t => t.definition.name -> t).toMap

  def dispatch(call: Part.FunctionCall): IO[Part.ToolOutput] =
    tools.get(call.name) match
      case Some(tool) =>
        tool.execute(call.args).map(output => Part.ToolOutput(call.name, output.value))
      case None =>
        IO.pure(Part.ToolOutput(call.name, s"Error: Tool '${call.name}' not found."))

object LiveGitAgent extends IOApp.Simple:

  // The main application entry point.
  override def run: IO[Unit] =
    // Use a cats-effect Resource to safely acquire and release the client.
    GenAiClient.geminiFromEnv().use {
      client =>
        for
          _ <- IO.println("─" * 50)
          _ <- IO.println("Gemini Git Agent Initialized.")
          _ <- IO.println("Enter your request (e.g., 'Check the git status') or 'exit'.")
          _ <- IO.print("> ")
          userQuery <- IO.readLine
          _ <- if userQuery.toLowerCase == "exit" then IO.unit else processQuery(client, userQuery)
        yield ()
    }

  // Processes a single user query.
  def processQuery(client: GenAiClient, userQuery: String): IO[Unit] =
    val systemPrompt = Content(Role.System, List(Part.Text(
      "You are a helpful assistant with access to local tools. Your purpose is to help users by executing git commands. When asked to do something with git, use the 'git' tool. Do not ask for permission. After getting the tool output, summarize it for the user."
    )))
    val userContent = Content(Role.User, List(Part.Text(userQuery)))
    val conversation = List(systemPrompt, userContent)
    val toolConfig = GenerationConfig(tools = List(GitTool.definition))

    for
      _ <- IO.println(s"Sending request to Gemini...")
      // First call to the LLM
      response <- client.generateContent(gemini20Flash, conversation, Some(toolConfig))

      // Check if the response contains a function call
      maybeFunctionCall = response.candidates.headOption.flatMap(_.content.parts.collectFirst {
        case fc: Part.FunctionCall => fc
      })

      _ <- maybeFunctionCall match
        // Case 1: The LLM wants to use a tool
        case Some(functionCall) =>
          for
            _      <- IO.println(s"LLM wants to run tool: ${functionCall.name} with args: ${functionCall.args}")
            toolOutput <- ToolDispatcher.dispatch(functionCall)
            _      <- IO.println("Tool executed. Sending result back to LLM...")

            // Append the function call and tool output to the conversation
            val newConversation = conversation ++ List(
              Content(Role.Model, List(functionCall)),
              Content(Role.User, List(toolOutput)) // In Gemini API, tool output is from user role
            )

            // Second call to the LLM to get a natural language summary
            finalResponse <- client.generateContent(gemini20Flash, newConversation, Some(toolConfig))
            _ <- printFinalResponse(finalResponse)
          yield ()

        // Case 2: The LLM responded directly with text
        case None =>
          printFinalResponse(response)

    yield ()

  // Helper to print the final text part from a response
  def printFinalResponse(response: com.google.genai.scala.types.Response): IO[Unit] =
    response.candidates.headOption.flatMap(_.content.parts.collectFirst {
      case Part.Text(text) => text
    }) match
      case Some(text) => IO.println("\n--- Gemini's Response ---") *> IO.println(text)
      case None => IO.println("Error: Received an empty response from the model.")
