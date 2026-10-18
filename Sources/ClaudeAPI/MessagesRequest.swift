// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Request body for `POST /v1/messages`.
package struct MessagesRequest: Sendable, Codable {
  package var model: String
  package var maxTokens: Int
  package var system: SystemPrompt?
  package var messages: [Message]
  package var tools: [ToolDefinition]?
  package var toolChoice: ToolChoice?
  package var thinking: ThinkingConfig?
  package var temperature: Double?
  package var topP: Double?
  package var topK: Int?
  package var cacheControl: CacheControl?
  package var outputConfig: OutputConfig?
  /// Substitute models for a policy refusal. Sending any needs the
  /// ``Fallbacks/betaHeader`` opt-in.
  package var fallbacks: Fallbacks?
  package var stream: Bool

  package init(
    model: String,
    maxTokens: Int = 16_000,
    system: SystemPrompt? = nil,
    messages: [Message],
    tools: [ToolDefinition]? = nil,
    toolChoice: ToolChoice? = nil,
    thinking: ThinkingConfig? = nil,
    temperature: Double? = nil,
    topP: Double? = nil,
    topK: Int? = nil,
    cacheControl: CacheControl? = nil,
    outputConfig: OutputConfig? = nil,
    fallbacks: Fallbacks? = nil,
    stream: Bool = false
  ) {
    self.model = model
    self.maxTokens = maxTokens
    self.system = system
    self.messages = messages
    self.tools = tools
    self.toolChoice = toolChoice
    self.thinking = thinking
    self.temperature = temperature
    self.topP = topP
    self.topK = topK
    self.cacheControl = cacheControl
    self.outputConfig = outputConfig
    self.fallbacks = fallbacks
    self.stream = stream
  }

  private enum CodingKeys: String, CodingKey {
    case model, system, messages, tools, thinking, stream, temperature, fallbacks
    case maxTokens = "max_tokens"
    case toolChoice = "tool_choice"
    case topP = "top_p"
    case topK = "top_k"
    case cacheControl = "cache_control"
    case outputConfig = "output_config"
  }
}

/// The `system` field: plain text, or one text block carrying a cache
/// breakpoint so the prompt can be reused across requests.
///
/// Request-level automatic caching puts its breakpoint on the last block. In
/// a single-turn session that's the new user message, so a shared system
/// prompt would be written on every request and never read. `.cached` gives
/// the system prompt its own breakpoint (one of the API's four).
package enum SystemPrompt: Sendable, Hashable, Codable, ExpressibleByStringLiteral {
  case text(String)
  case cached(String, CacheControl)

  /// The prompt text, whichever way it is sent.
  package var text: String {
    switch self {
    case .text(let t), .cached(let t, _): t
    }
  }

  package init(stringLiteral value: String) { self = .text(value) }

  private struct Block: Codable {
    var type = "text"
    var text: String
    var cacheControl: CacheControl?

    private enum CodingKeys: String, CodingKey {
      case type, text
      case cacheControl = "cache_control"
    }
  }

  package init(from decoder: Decoder) throws {
    let c = try decoder.singleValueContainer()
    if let text = try? c.decode(String.self) {
      self = .text(text)
      return
    }
    let blocks = try c.decode([Block].self)
    let text = blocks.map(\.text).joined(separator: "\n\n")
    if let cache = blocks.last?.cacheControl {
      self = .cached(text, cache)
    } else {
      self = .text(text)
    }
  }

  package func encode(to encoder: Encoder) throws {
    var c = encoder.singleValueContainer()
    switch self {
    case .text(let text):
      try c.encode(text)
    case .cached(let text, let cache):
      try c.encode([Block(text: text, cacheControl: cache)])
    }
  }
}

/// Output shaping: structured output via constrained decoding, effort level.
package struct OutputConfig: Sendable, Hashable, Codable {
  package var format: Format?
  package var effort: Effort?

  package init(format: Format? = nil, effort: Effort? = nil) {
    self.format = format
    self.effort = effort
  }

  /// Constrained-decoding output format. The model strictly conforms to the
  /// schema — it cannot emit a token that would violate it.
  package struct Format: Sendable, Hashable, Codable {
    package var schema: JSONValue

    package init(schema: JSONValue) { self.schema = schema }

    private enum CodingKeys: String, CodingKey { case type, schema }

    package init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      schema = try c.decode(JSONValue.self, forKey: .schema)
    }

    package func encode(to encoder: Encoder) throws {
      var c = encoder.container(keyedBy: CodingKeys.self)
      try c.encode("json_schema", forKey: .type)
      try c.encode(schema, forKey: .schema)
    }
  }

  package enum Effort: String, Sendable, Codable {
    case low, medium, high, xhigh, max
  }
}
