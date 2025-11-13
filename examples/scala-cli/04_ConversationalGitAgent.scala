//> using scala "3.3.1"
//> using dep "org.typelevel::cats-effect::3.5.4"
//> using dep "co.fs2::fs2-core::3.10.2"
//> using dep "io.circe::circe-core::0.14.7"
//> using dep "io.circe::circe-generic::0.14.7"
//> using dep "io.circe::circe-parser::0.14.7"

// This is a HYPOTHETICAL dependency. It does not exist.
//> using dep "com.google.genai::genai-scala-iron::0.1.0-SNAPSHOT"

import cats.effect.{IO, IOApp, ExitCode}
import cats.syntax.all.*
import scala.sys.process.*
import scala.util.Try

// --- Hypothetical imports from the scala-wrapper ---
import com.google.genai.scala.client.GenAiClient
import com.google.genai.scala.types.{Content, Part, Role, GenerationConfig, Tool => GenAiTool}
import com.google.genai.scala.types.Model.gemini20Flash

// --- Tool Definition (Identical to 03_LiveGitAgent.scala) ---
trait Tool:
  def definition: GenAiTool
  def execute(args: Map[String, String]): IO[String]

object GitTool extends Tool:
  import com.google.genai.scala.types.Schema
  override val definition: GenAiTool = GenAiTool(
    name = "git",
    description = "Executes a git command locally. The input is a single string containing all arguments for the git command.",
    schema = Schema.obj(properties = Map("args" -> Schema.string("The arguments to pass to the git command.")), required = List("args"))
  )
  override def execute(args: Map[String, String]): IO[String] = IO.blocking {
    args.get("args").map {
      rawArgs =>
        val command = Seq("git") ++ rawArgs.split(' ').filter(_.nonEmpty)
        val stdout = new StringBuilder
        val stderr = new StringBuilder
        command.!(ProcessLogger(stdout.append(_).append("\n"), stderr.append(_).append("\n")))
        s"STDOUT:\n${stdout.toString}\nSTDERR:\n${stderr.toString}"
    }.getOrElse("Error: Missing required 'args' parameter.")
  }

// --- Agent Components (Identical to 03_LiveGitAgent.scala) ---
object ToolDispatcher:
  private val tools: Map[String, Tool] = Map(GitTool.definition.name -> GitTool)
  def dispatch(call: Part.FunctionCall): IO[Part.ToolOutput] =
    tools.get(call.name) match
      case Some(tool) => tool.execute(call.args).map(Part.ToolOutput(call.name, _))
      case None => IO.pure(Part.ToolOutput(call.name, s"Error: Tool '${call.name}' not found."))

object ConversationalGitAgent extends IOApp.Simple:

  // The main application entry point.
  override def run: IO[Unit] =
    GenAiClient.geminiFromEnv().use {
      client =>
        val systemPrompt = Content(Role.System, List(Part.Text(
          "You are a helpful assistant with access to local tools. Your purpose is to help users by executing git commands. When asked to do something with git, use the 'git' tool. After getting the tool output, summarize it for the user and await their next instruction."
        )))
        for
          _ <- IO.println("─" * 50)
          _ <- IO.println("Gemini Conversational Git Agent Initialized.")
          _ <- IO.println("Enter your request or 'exit'.")
          // Start the recursive conversation loop with the initial system prompt.
          _ <- conversationLoop(client, List(systemPrompt))
        yield ()
    }

  // A recursive loop that maintains conversation history.
  def conversationLoop(client: GenAiClient, history: List[Content]): IO[Unit] =
    for
      _ <- IO.print("> ")
      userInput <- IO.readLine
      _ <- if userInput.toLowerCase == "exit" then IO.println("Goodbye!") else {
        val userContent = Content(Role.User, List(Part.Text(userInput)))
        val currentHistory = history :+ userContent
        val toolConfig = GenerationConfig(tools = List(GitTool.definition))

        for
          _ <- IO.println("Sending request to Gemini...")
          response <- client.generateContent(gemini20Flash, currentHistory, Some(toolConfig))

          // Extract the first response from the model.
          responseContent = response.candidates.headOption.map(_.content)
          
          // Check if the response contains a function call.
          maybeFunctionCall = responseContent.flatMap(_.parts.collectFirst { case fc: Part.FunctionCall => fc })

          // Process the response and get the new history for the next loop iteration.
          newHistory <- maybeFunctionCall match
            case Some(functionCall) => handleFunctionCall(client, toolConfig, currentHistory, functionCall)
            case None => handleTextResponse(currentHistory, responseContent)
          
          // Continue the loop with the updated history.
          _ <- conversationLoop(client, newHistory)
        yield ()
      }
    yield ()

  // Handles the case where the LLM returns a function call.
  def handleFunctionCall(client: GenAiClient, toolConfig: GenerationConfig, history: List[Content], call: Part.FunctionCall): IO[List[Content]] =
    for
      _ <- IO.println(s"LLM wants to run tool: ${call.name} with args: ${call.args}")
      toolOutput <- ToolDispatcher.dispatch(call)
      _ <- IO.println("Tool executed. Sending result back to LLM...")

      // The history now includes the model's request to use a tool and the tool's output.
      historyWithToolOutput = history ++ List(
        Content(Role.Model, List(call)),
        Content(Role.User, List(toolOutput)) // Tool output is from the 'user' role
      )

      // Make a second call to get the LLM's summary.
      finalResponse <- client.generateContent(gemini20Flash, historyWithToolOutput, Some(toolConfig))
      finalContent = finalResponse.candidates.headOption.map(_.content)
      _ <- printFinalText(finalContent)
    yield historyWithToolOutput ++ finalContent.toList // Add LLM's summary to history

  // Handles the case where the LLM returns a simple text response.
  def handleTextResponse(history: List[Content], responseContent: Option[Content]): IO[List[Content]] =
    for
      _ <- printFinalText(responseContent)
    yield history ++ responseContent.toList // Add LLM's response to history

  // Helper to print the final text part from a response content.
  def printFinalText(responseContent: Option[Content]): IO[Unit] =
    responseContent.flatMap(_.parts.collectFirst { case Part.Text(text) => text }) match
      case Some(text) => IO.println("\n---" * 10) *> IO.println(text) *> IO.println("--------------")
      case None => IO.println("Warning: Received an empty text response from the model.")
