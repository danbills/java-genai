//> using scala "3.3.1"

import scala.sys.process.*
import scala.util.{Try, Success, Failure}

/**
 * This script defines the foundational components for a tool-using agent.
 * - It uses opaque types for strong, zero-cost type safety.
 * - It defines a generic `Tool` trait.
 * - It provides a concrete `GitTool` implementation for executing local git commands.
 * - A `main` method demonstrates how to use the tool.
 */

// --- Strong, Semantic Types ---
// Using opaque types to provide semantic meaning to primitive types like String
// without any runtime overhead. The Scala 3 compiler will enforce these types.
object Types:
  opaque type ToolName = String
  object ToolName:
    def apply(name: String): ToolName = name
  extension (name: ToolName) def value: String = name

  opaque type ToolDescription = String
  object ToolDescription:
    def apply(desc: String): ToolDescription = desc

  // A placeholder for the rich Schema type that would be provided by the scala-wrapper.
  // For this example, we'll just use a String containing a JSON schema definition.
  opaque type JsonSchema = String
  object JsonSchema:
    def apply(schema: String): JsonSchema = schema
  extension (schema: JsonSchema) def value: String = schema

  opaque type GitArgs = String
  object GitArgs:
    def apply(args: String): GitArgs = args
  extension (args: GitArgs) def value: String = args

  opaque type ToolOutput = String
  object ToolOutput:
    def apply(output: String): ToolOutput = output
  extension (output: ToolOutput) def value: String = output

import Types.*

// --- Tool Definition ---

/**
 * A generic trait representing a tool that the AI agent can use.
 */
trait Tool:
  def name: ToolName
  def description: ToolDescription
  def schema: JsonSchema
  def execute(args: Map[String, String]): Either[String, ToolOutput]

// --- Git Tool Implementation ---

/**
 * An implementation of `Tool` for executing local `git` commands.
 */
object GitTool extends Tool:
  override val name: ToolName = ToolName("git")
  override val description: ToolDescription = ToolDescription(
    "Executes a git command locally. The input is a single string containing all arguments for the git command (e.g., 'status -s' or 'log --oneline -n 5')."
  )
  override val schema: JsonSchema = JsonSchema(
    """|{
       |  "type": "object",
       |  "properties": {
       |    "args": {
       |      "type": "string",
       |      "description": "The arguments to pass to the git command."
       |    }
       |  },
       |  "required": ["args"]
       |}"""
  )

  override def execute(args: Map[String, String]): Either[String, ToolOutput] =
    args.get("args") match
      case Some(rawArgs) =>
        val gitArgs = GitArgs(rawArgs)
        // SECURITY WARNING: Executing arbitrary shell commands from an LLM is dangerous.
        // This example is for demonstration in a controlled, local environment.
        // In a real application, you must strictly sanitize inputs and limit commands.
        Try {
          val command = Seq("git") ++ gitArgs.value.split(' ').filter(_.nonEmpty)
          val stdout = new StringBuilder
          val stderr = new StringBuilder
          val logger = ProcessLogger(stdout.append(_).append("\n"), stderr.append(_).append("\n"))
          val status = command.!(logger)

          val output = s"Status: $status\n\nSTDOUT:\n${stdout.toString}\nSTDERR:\n${stderr.toString}"
          ToolOutput(output)
        } match
          case Success(output) => Right(output)
          case Failure(e) => Left(s"Failed to execute git command: ${e.getMessage}")

      case None => Left("Error: Missing required 'args' parameter for the git tool.")

// --- Main Application ---
@main def runToolDefinition(): Unit =
  println("--- Tool Definition ---")
  println(s"Name: ${GitTool.name.value}")
  println(s"Description: ${GitTool.description}")
  println(s"Schema:\n${GitTool.schema.value}")
  println("-" * 20)

  println("\n--- Use Case 1: Successful Command ---")
  val successArgs = Map("args" -> "status --short")
  println(s"Executing with args: $successArgs")
  GitTool.execute(successArgs) match
    case Right(output) => println(s"Result:\n${output.value}")
    case Left(error) => println(s"Error: $error")
  println("-" * 20)

  println("\n--- Use Case 2: Command with Error ---")
  val errorArgs = Map("args" -> "invalid-command")
  println(s"Executing with args: $errorArgs")
  GitTool.execute(errorArgs) match
    case Right(output) => println(s"Result:\n${output.value}")
    case Left(error) => println(s"Error: $error")
  println("-" * 20)

  println("\n--- Use Case 3: Missing Arguments ---")
  println(s"Executing with empty args: {}")
  GitTool.execute(Map.empty) match
    case Right(output) => println(s"Result:\n${output.value}")
    case Left(error) => println(s"Error: $error")
  println("-" * 20)
