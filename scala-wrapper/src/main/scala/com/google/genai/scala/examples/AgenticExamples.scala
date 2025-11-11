package com.google.genai.scala.examples

import cats.effect.*
import cats.syntax.all.*
import com.google.genai.scala.client.*
import com.google.genai.scala.types.*
import com.google.genai.scala.types.Model.*
import com.google.genai.scala.constraints.StringConstraints.*
import com.google.genai.scala.agent.*
import com.google.genai.scala.agent.FunctionExecutorDSL.*

/**
 * Examples demonstrating agentic behavior with function calling.
 *
 * Shows:
 * 1. Simple function calling
 * 2. Multi-turn agentic loops
 * 3. Tool chaining
 * 4. Error handling
 * 5. Stateful agents
 */
object AgenticExamples extends IOApp.Simple:

  def run: IO[Unit] =
    GenAiClient.geminiFromEnv().use { client =>
      for
        _ <- IO.println("=== Agentic Function Calling Examples ===\n")
        _ <- example1_SimpleCalculator(client)
        _ <- example2_WeatherAgent(client)
        _ <- example3_DatabaseAgent(client)
        _ <- example4_MultiToolAgent(client)
        _ <- example5_StatefulAgent(client)
      yield ()
    }

  // ============================================================================
  // EXAMPLE 1: Simple Calculator
  // ============================================================================

  def example1_SimpleCalculator(client: GeminiApiClient): IO[Unit] =
    IO.println("--- Example 1: Simple Calculator ---") *> {

      // Define calculator functions
      val addFunction = FunctionDeclaration(
        name = "add",
        description = "Add two numbers together and return the sum",
        FunctionParameter.integer("a", "First number to add"),
        FunctionParameter.integer("b", "Second number to add")
      )

      val multiplyFunction = FunctionDeclaration(
        name = "multiply",
        description = "Multiply two numbers together and return the product",
        FunctionParameter.integer("a", "First number to multiply"),
        FunctionParameter.integer("b", "Second number to multiply")
      )

      // Build executor with implementations
      val executor = builder
        .registerSimple(addFunction, args =>
          IO {
            val a = args("a").toString.toInt
            val b = args("b").toString.toInt
            val result = a + b
            s"The sum of $a and $b is $result"
          }
        )
        .registerSimple(multiplyFunction, args =>
          IO {
            val a = args("a").toString.toInt
            val b = args("b").toString.toInt
            val result = a * b
            s"The product of $a and $b is $result"
          }
        )
        .build

      // Create agentic executor
      val agent = AgenticExecutor(client, executor)

      // Run calculation
      val prompt = Prompt.unsafe("What is (12 + 8) multiplied by 5?")

      agent.executeLoop(
        initialPrompt = prompt,
        model = gemini20Flash,
        config = GenerationConfig.balanced,
        afcConfig = AutomaticFunctionCallingConfig(
          maxIterations = MaxIterations.unsafe(10),
          verbose = true
        )
      ).flatMap { result =>
        IO.println(s"\nFinal Answer: ${result.finalText.getOrElse("No response")}") *>
        IO.println(s"Function Calls Made: ${result.functionCallCount}") *>
        IO.println(s"Iterations: ${result.iterationCount}\n")
      }
    }

  // ============================================================================
  // EXAMPLE 2: Weather Agent
  // ============================================================================

  def example2_WeatherAgent(client: GeminiApiClient): IO[Unit] =
    IO.println("--- Example 2: Weather Agent ---") *> {

      // Simulated weather database
      val weatherData = Map(
        "new york" -> (72, "Partly cloudy"),
        "london" -> (61, "Rainy"),
        "tokyo" -> (78, "Clear"),
        "sydney" -> (68, "Sunny")
      )

      // Define weather functions
      val getCurrentWeatherFunction = FunctionDeclaration(
        name = "get_current_weather",
        description = "Get the current weather for a specified city",
        FunctionParameter.string("city", "The city name to get weather for"),
        FunctionParameter.enum("unit", "Temperature unit (celsius or fahrenheit)",
          List("celsius", "fahrenheit"))
      )

      val getWeatherForecastFunction = FunctionDeclaration(
        name = "get_weather_forecast",
        description = "Get weather forecast for the next N days",
        FunctionParameter.string("city", "The city name to get forecast for"),
        FunctionParameter.integer("days", "Number of days to forecast", min = Some(1), max = Some(7))
      )

      // Build executor
      val executor = builder
        .registerSimple(getCurrentWeatherFunction, args =>
          IO {
            val city = args("city").toString.toLowerCase
            val unit = args("unit").toString

            weatherData.get(city) match
              case Some((temp, condition)) =>
                val displayTemp = if unit == "celsius" then ((temp - 32) * 5 / 9) else temp
                s"Current weather in $city: $displayTemp°${if unit == "celsius" then "C" else "F"}, $condition"
              case None =>
                s"Sorry, I don't have weather data for $city"
          }
        )
        .registerSimple(getWeatherForecastFunction, args =>
          IO {
            val city = args("city").toString.toLowerCase
            val days = args("days").toString.toInt

            weatherData.get(city) match
              case Some((baseTemp, _)) =>
                val forecast = (1 to days).map { day =>
                  val temp = baseTemp + (scala.util.Random.nextInt(10) - 5)
                  s"Day $day: $temp°F"
                }.mkString(", ")
                s"$days-day forecast for $city: $forecast"
              case None =>
                s"Sorry, I don't have forecast data for $city"
          }
        )
        .build

      // Create agent
      val agent = AgenticExecutor(client, executor)

      // Ask complex weather question
      val prompt = Prompt.unsafe(
        "What's the weather like in London and Tokyo? Also, give me a 3-day forecast for New York."
      )

      agent.executeLoop(
        initialPrompt = prompt,
        model = gemini20Flash,
        afcConfig = AutomaticFunctionCallingConfig(verbose = true)
      ).flatMap { result =>
        IO.println(s"\nAgent Response:\n${result.finalText.getOrElse("No response")}") *>
        IO.println(s"\nFunction Calls: ${result.functionCallCount}\n")
      }
    }

  // ============================================================================
  // EXAMPLE 3: Database Agent
  // ============================================================================

  def example3_DatabaseAgent(client: GeminiApiClient): IO[Unit] =
    IO.println("--- Example 3: Database Agent ---") *> {

      // Simulated database
      case class User(id: Int, name: String, email: String, role: String)

      val users = Map(
        1 -> User(1, "Alice Smith", "alice@example.com", "admin"),
        2 -> User(2, "Bob Jones", "bob@example.com", "user"),
        3 -> User(3, "Charlie Brown", "charlie@example.com", "moderator")
      )

      // Define database functions
      val getUserFunction = FunctionDeclaration(
        name = "get_user",
        description = "Retrieve user information by user ID",
        FunctionParameter.integer("user_id", "The ID of the user to retrieve", min = Some(1))
      )

      val searchUsersFunction = FunctionDeclaration(
        name = "search_users",
        description = "Search for users by name or email",
        FunctionParameter.string("query", "Search query to match against name or email")
      )

      val listUsersByRoleFunction = FunctionDeclaration(
        name = "list_users_by_role",
        description = "List all users with a specific role",
        FunctionParameter.enum("role", "User role to filter by",
          List("admin", "user", "moderator"))
      )

      // Build executor
      val executor = builder
        .register(getUserFunction, args =>
          IO {
            val userId = args("user_id").toString.toInt
            users.get(userId) match
              case Some(user) =>
                FunctionResult.success(Map(
                  "id" -> user.id,
                  "name" -> user.name,
                  "email" -> user.email,
                  "role" -> user.role
                ))
              case None =>
                FunctionResult.error(s"User with ID $userId not found")
          }
        )
        .registerSimple(searchUsersFunction, args =>
          IO {
            val query = args("query").toString.toLowerCase
            val matches = users.values.filter { user =>
              user.name.toLowerCase.contains(query) ||
              user.email.toLowerCase.contains(query)
            }

            if matches.isEmpty then
              s"No users found matching '$query'"
            else
              matches.map(u => s"${u.name} (${u.email})").mkString(", ")
          }
        )
        .registerSimple(listUsersByRoleFunction, args =>
          IO {
            val role = args("role").toString
            val matches = users.values.filter(_.role == role)

            if matches.isEmpty then
              s"No users with role '$role'"
            else
              matches.map(u => s"${u.name} (ID: ${u.id})").mkString(", ")
          }
        )
        .build

      // Create agent
      val agent = AgenticExecutor(client, executor)

      // Ask complex database question
      val prompt = Prompt.unsafe(
        "Who are all the admins? Also, find me the user with 'Bob' in their name and tell me their role."
      )

      agent.executeLoop(
        initialPrompt = prompt,
        model = gemini20Flash,
        afcConfig = AutomaticFunctionCallingConfig(verbose = true)
      ).flatMap { result =>
        IO.println(s"\nAgent Response:\n${result.finalText.getOrElse("No response")}\n")
      }
    }

  // ============================================================================
  // EXAMPLE 4: Multi-Tool Agent (Combines Multiple Tools)
  // ============================================================================

  def example4_MultiToolAgent(client: GeminiApiClient): IO[Unit] =
    IO.println("--- Example 4: Multi-Tool Agent ---") *> {

      // Define diverse set of tools
      val getTimeFunction = FunctionDeclaration(
        name = "get_current_time",
        description = "Get the current time in a specific timezone",
        FunctionParameter.string("timezone", "Timezone name (e.g., 'America/New_York', 'Europe/London')")
      )

      val convertCurrencyFunction = FunctionDeclaration(
        name = "convert_currency",
        description = "Convert an amount from one currency to another",
        FunctionParameter.integer("amount", "Amount to convert", min = Some(0)),
        FunctionParameter.string("from_currency", "Source currency code (USD, EUR, GBP, etc.)"),
        FunctionParameter.string("to_currency", "Target currency code")
      )

      val translateTextFunction = FunctionDeclaration(
        name = "translate_text",
        description = "Translate text from one language to another",
        FunctionParameter.string("text", "Text to translate"),
        FunctionParameter.string("from_lang", "Source language code"),
        FunctionParameter.string("to_lang", "Target language code")
      )

      // Build executor
      val executor = builder
        .registerSimple(getTimeFunction, args =>
          IO {
            val timezone = args("timezone").toString
            val time = java.time.ZonedDateTime.now(java.time.ZoneId.of(timezone))
            s"Current time in $timezone: ${time.format(java.time.format.DateTimeFormatter.RFC_1123_DATE_TIME)}"
          }
        )
        .registerSimple(convertCurrencyFunction, args =>
          IO {
            val amount = args("amount").toString.toInt
            val from = args("from_currency").toString.toUpperCase
            val to = args("to_currency").toString.toUpperCase

            // Simulated exchange rates
            val rates = Map("USD" -> 1.0, "EUR" -> 0.92, "GBP" -> 0.79, "JPY" -> 149.5)

            (rates.get(from), rates.get(to)) match
              case (Some(fromRate), Some(toRate)) =>
                val result = (amount / fromRate) * toRate
                f"$amount $from = $result%.2f $to"
              case _ =>
                s"Unsupported currency conversion: $from to $to"
          }
        )
        .registerSimple(translateTextFunction, args =>
          IO {
            val text = args("text").toString
            val fromLang = args("from_lang").toString
            val toLang = args("to_lang").toString

            // Simulated translation (in reality, would call translation API)
            s"[Simulated translation from $fromLang to $toLang]: $text"
          }
        )
        .build

      // Create agent
      val agent = AgenticExecutor(client, executor)

      // Complex multi-tool query
      val prompt = Prompt.unsafe(
        """What time is it in Tokyo right now?
           Also, if I have 100 USD, how much is that in EUR and GBP?"""
      )

      agent.executeLoop(
        initialPrompt = prompt,
        model = gemini20Flash,
        afcConfig = AutomaticFunctionCallingConfig(verbose = true)
      ).flatMap { result =>
        IO.println(s"\nAgent Response:\n${result.finalText.getOrElse("No response")}\n")
      }
    }

  // ============================================================================
  // EXAMPLE 5: Stateful Agent (Maintains State Across Calls)
  // ============================================================================

  def example5_StatefulAgent(client: GeminiApiClient): IO[Unit] =
    IO.println("--- Example 5: Stateful Shopping Cart Agent ---") *> {

      // Shopping cart state
      case class CartItem(name: String, price: Double, quantity: Int)
      val cart = scala.collection.mutable.Map[String, CartItem]()

      // Define shopping functions
      val addToCartFunction = FunctionDeclaration(
        name = "add_to_cart",
        description = "Add an item to the shopping cart",
        FunctionParameter.string("item_name", "Name of the item to add"),
        FunctionParameter.integer("quantity", "Quantity to add", min = Some(1)),
        FunctionParameter.integer("price", "Price per item in cents", min = Some(0))
      )

      val removeFromCartFunction = FunctionDeclaration(
        name = "remove_from_cart",
        description = "Remove an item from the shopping cart",
        FunctionParameter.string("item_name", "Name of the item to remove")
      )

      val viewCartFunction = FunctionDeclaration(
        name = "view_cart",
        description = "View all items currently in the shopping cart with total price"
      )

      val checkoutFunction = FunctionDeclaration(
        name = "checkout",
        description = "Complete the purchase and clear the cart"
      )

      // Build executor
      val executor = builder
        .registerSimple(addToCartFunction, args =>
          IO {
            val name = args("item_name").toString
            val quantity = args("quantity").toString.toInt
            val price = args("price").toString.toInt / 100.0

            cart.get(name) match
              case Some(existing) =>
                cart(name) = existing.copy(quantity = existing.quantity + quantity)
                s"Updated $name quantity to ${existing.quantity + quantity}"
              case None =>
                cart(name) = CartItem(name, price, quantity)
                s"Added $quantity x $name to cart at $$${price} each"
          }
        )
        .registerSimple(removeFromCartFunction, args =>
          IO {
            val name = args("item_name").toString
            cart.remove(name) match
              case Some(_) => s"Removed $name from cart"
              case None => s"$name not found in cart"
          }
        )
        .registerSimple(viewCartFunction, _ =>
          IO {
            if cart.isEmpty then
              "Cart is empty"
            else
              val items = cart.values.map { item =>
                val itemTotal = item.price * item.quantity
                f"- ${item.name}: ${item.quantity} x $$${item.price}%.2f = $$${itemTotal}%.2f"
              }.mkString("\n")

              val total = cart.values.map(i => i.price * i.quantity).sum
              s"Shopping Cart:\n$items\n\nTotal: $$${total}%.2f"
          }
        )
        .registerSimple(checkoutFunction, _ =>
          IO {
            val total = cart.values.map(i => i.price * i.quantity).sum
            val itemCount = cart.values.map(_.quantity).sum
            cart.clear()
            f"Checkout complete! Purchased $itemCount items for $$${total}%.2f. Cart is now empty."
          }
        )
        .build

      // Create agent
      val agent = AgenticExecutor(client, executor)

      // Interactive shopping session
      for
        _ <- IO.println("Starting shopping session...")

        // First interaction: add items
        result1 <- agent.executeLoop(
          initialPrompt = Prompt.unsafe("Add 2 apples at $1.50 each and 3 oranges at $2.00 each to my cart"),
          model = gemini20Flash,
          afcConfig = AutomaticFunctionCallingConfig(verbose = false)
        )
        _ <- IO.println(s"Assistant: ${result1.finalText.getOrElse("No response")}\n")

        // Second interaction: view cart (using chat to maintain conversation)
        result2 <- agent.chat(
          conversation = result1.conversation,
          userMessage = Prompt.unsafe("What's in my cart now?"),
          model = gemini20Flash,
          afcConfig = AutomaticFunctionCallingConfig(verbose = false)
        )
        _ <- IO.println(s"Assistant: ${result2.finalText.getOrElse("No response")}\n")

        // Third interaction: checkout
        result3 <- agent.chat(
          conversation = result2.conversation,
          userMessage = Prompt.unsafe("Looks good, let's checkout!"),
          model = gemini20Flash,
          afcConfig = AutomaticFunctionCallingConfig(verbose = false)
        )
        _ <- IO.println(s"Assistant: ${result3.finalText.getOrElse("No response")}\n")

      yield ()
    }
