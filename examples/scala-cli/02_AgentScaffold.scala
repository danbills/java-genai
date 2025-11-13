//> using scala "3.3.1"

import scala.sys.process.*
import scala.util.{Try, Success, Failure}

/**
 * This script scaffolds the agent's core logic.
 * - It simulates a function call from the LLM.
 * - It implements a `ToolDispatcher` to route the simulated call to the correct tool.
 * - This tests the agent's internal wiring without making live API calls.
 */

// --- Strong, Semantic Types (Copied from 01_ToolDefinition.scala) ---
object Types:
  opaque type ToolName = String
  object ToolName:
    def apply(name: String): ToolName = name
  extension (name: ToolName) def value: String = name

  opaque type ToolDescription = String
  object ToolDescription:
    def apply(desc: String): ToolDescription = desc

  opaque type JsonSchema = String
  object JsonSchema:
    def apply(schema: String): JsonSchema = schema
  extension (schema: JsonSchema) def value: String = schema

  opaque type ToolOutput = String
  object ToolOutput:
    def apply(output: String): ToolOutput = output
  extension (output: ToolOutput) def value: String = output
import Types.*

// --- Tool Definition (Copied from 01_ToolDefinition.scala) ---
trait Tool:
  def name: ToolName
  def description: ToolDescription
  def schema: JsonSchema
  def execute(args: Map[String, String]): Either[String, ToolOutput]

object GitTool extends Tool:
  override val name: ToolName = ToolName("git")
  override val description: ToolDescription = ToolDescription(
    "Executes a git command locally. The input is a single string containing all arguments for the git command (e.g., 'status -s' or 'log --oneline -n 5')."
  )
  override val schema: JsonSchema = JsonSchema(
    """{"type": "object", "properties": {"args": {"type": "string"}}, "required": ["args"]}"""
  )

  override def execute(args: Map[String, String]): Either[String, ToolOutput] =
    args.get("args") match
      case Some(rawArgs) =>
        Try {
          val command = Seq("git") ++ rawArgs.split(' ').filter(_.nonEmpty)
          val stdout = new StringBuilder
          val stderr = new StringBuilder
          val logger = ProcessLogger(stdout.append(_).append("\n"), stderr.append(_).append("\n"))
          command.!(logger)
          ToolOutput(s"STDOUT:\n${stdout.toString}\nSTDERR:\n${stderr.toString}")
        }.toEither.left.map(e => s"Failed to execute git command: ${e.getMessage}")
      case None => Left("Error: Missing required 'args' parameter for the git tool.")

// --- Agent Components ---

/**
 * A placeholder for the `FunctionCall` type from the scala-wrapper.
 */
case class FunctionCall(name: String, args: Map[String, String])

/**
 * A dispatcher that holds a registry of available tools and executes them.
 */
object ToolDispatcher:
  // A registry of all tools the agent can use.
  private val tools: Map[ToolName, Tool] = List(GitTool).map(t => t.name -> t).toMap

  def dispatch(call: FunctionCall): Either[String, ToolOutput] =
    tools.get(ToolName(call.name)) match
      case Some(tool) => tool.execute(call.args)
      case None => Left(s"Error: Tool '${call.name}' not found.")

// --- Main Application ---
@main def runAgentScaffold(): Unit =
  println("--- Agent Scaffold Simulation ---")

  val userQuery = "What are the last 2 commits?"
  println(s"User Query: '$userQuery'")
  println("-" * 20)

  // Simulate the LLM's response: a call to the 'git' tool.
  val simulatedFunctionCall = FunctionCall(
    name = "git",
    args = Map("args" -> "log --oneline -n 2")
  )
  println(s"Simulated LLM Response (FunctionCall): $simulatedFunctionCall")
  println("-" * 20)

  println("Dispatching to tool...")
  // The dispatcher finds the right tool and executes it.
  ToolDispatcher.dispatch(simulatedFunctionCall) match
    case Right(result) =>
      println("Tool Execution Succeeded!")
      println("Result:")
      println(result.value)
    case Left(error) =>
      println(s"Tool Execution Failed: $error")

  println("-" * 20)
