name := "genai-scala-iron"
version := "0.1.0-SNAPSHOT"
scalaVersion := "3.3.1"

libraryDependencies ++= Seq(
  // Iron for refinement types
  "io.github.iltotore" %% "iron" % "3.2.1",
  "io.github.iltotore" %% "iron-circe" % "3.2.1",

  // Circe for JSON
  "io.circe" %% "circe-core" % "0.14.6",
  "io.circe" %% "circe-generic" % "0.14.6",
  "io.circe" %% "circe-parser" % "0.14.6",

  // Cats for functional programming
  "org.typelevel" %% "cats-core" % "2.10.0",
  "org.typelevel" %% "cats-effect" % "3.5.2",

  // Http4s for HTTP client
  "org.http4s" %% "http4s-client" % "0.23.24",
  "org.http4s" %% "http4s-circe" % "0.23.24",
  "org.http4s" %% "http4s-ember-client" % "0.23.24",

  // FS2 for streaming
  "co.fs2" %% "fs2-core" % "3.9.3",

  // Testing
  "org.scalameta" %% "munit" % "0.7.29" % Test,
  "org.typelevel" %% "munit-cats-effect-3" % "1.0.7" % Test
)

scalacOptions ++= Seq(
  "-encoding", "UTF-8",
  "-feature",
  "-language:implicitConversions",
  "-unchecked",
  "-Xfatal-warnings",
  "-Ykind-projector",
  "-explain"
)
