// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import ClaudeAPI
import Foundation
import FoundationModels
import Testing

@testable import ClaudeForFoundationModels

@Suite struct ClaudeCacheDiagnosticsTests {
  private func response(id: String, diagnostics: String = "null") throws -> MessagesResponse {
    let json = """
      {"id":"\(id)","model":"claude-a","role":"assistant","content":[],"stop_reason":null,
       "usage":{"input_tokens":7,"output_tokens":1,"cache_read_input_tokens":500,
                "cache_creation_input_tokens":30},
       "diagnostics":\(diagnostics)}
      """
    return try JSONDecoder().decode(MessagesResponse.self, from: Data(json.utf8))
  }

  @Test func `the first request opts in with null, and each later one names the one before`()
    throws
  {
    let diagnostics = ClaudeCacheDiagnostics()
    #expect(diagnostics.nextRequest == .init(previousMessageID: nil))
    diagnostics.record(try response(id: "msg_1"))
    #expect(diagnostics.nextRequest == .init(previousMessageID: "msg_1"))
    diagnostics.record(try response(id: "msg_2"))
    #expect(diagnostics.nextRequest == .init(previousMessageID: "msg_2"))
  }

  // The API answers `null` to a first turn and to a clean comparison alike.
  @Test func `null reads as first request or no divergence by whether it was compared`() throws {
    let diagnostics = ClaudeCacheDiagnostics()
    diagnostics.record(try response(id: "msg_1"))
    diagnostics.record(try response(id: "msg_2"))
    #expect(diagnostics.takeReports().map(\.outcome) == [.firstRequest, .noDivergence])
  }

  @Test func `reports the API's reason, its estimate, and the prompt-side usage`() throws {
    let diagnostics = ClaudeCacheDiagnostics()
    diagnostics.record(try response(id: "msg_1"))
    diagnostics.record(
      try response(
        id: "msg_2",
        diagnostics: #"{"cache_miss_reason":{"type":"system_changed","cache_missed_input_tokens":41850}}"#
      )
    )
    let report = try #require(diagnostics.takeReports().last)
    #expect(report.outcome == .cacheMiss(reason: "system_changed", missedInputTokens: 41850))
    #expect(report.messageID == "msg_2")
    #expect(report.previousMessageID == "msg_1")
    #expect(report.inputTokens == 7)
    #expect(report.cacheReadInputTokens == 500)
    #expect(report.cacheCreationInputTokens == 30)
  }

  @Test func `a reason without an estimate, and an unknown reason, still come through`() throws {
    let diagnostics = ClaudeCacheDiagnostics()
    diagnostics.record(try response(id: "msg_1"))
    diagnostics.record(
      try response(
        id: "msg_2",
        diagnostics: #"{"cache_miss_reason":{"type":"previous_message_not_found"}}"#
      )
    )
    diagnostics.record(
      try response(id: "msg_3", diagnostics: #"{"cache_miss_reason":{"type":"a_new_reason"}}"#)
    )
    #expect(
      diagnostics.takeReports().dropFirst().map(\.outcome) == [
        .cacheMiss(reason: "previous_message_not_found", missedInputTokens: nil),
        .cacheMiss(reason: "a_new_reason", missedInputTokens: nil),
      ]
    )
  }

  // Diagnostics are best-effort, so no shape of the field may fail a response.
  @Test func `a pending comparison and an unknown shape are inconclusive`() throws {
    let diagnostics = ClaudeCacheDiagnostics()
    diagnostics.record(try response(id: "msg_1"))
    diagnostics.record(try response(id: "msg_2", diagnostics: #"{"cache_miss_reason":null}"#))
    diagnostics.record(try response(id: "msg_3", diagnostics: #""not an object""#))
    #expect(diagnostics.takeReports().dropFirst().map(\.outcome) == [.pending, .pending])
  }

  @Test func `taking reports empties them and keeps the chain`() throws {
    let diagnostics = ClaudeCacheDiagnostics()
    diagnostics.record(try response(id: "msg_1"))
    #expect(diagnostics.takeReports().count == 1)
    #expect(diagnostics.takeReports().isEmpty)
    #expect(diagnostics.nextRequest == .init(previousMessageID: "msg_1"))
  }

  // Each session is fresh, as Robi's are: the chain lives on the instance.
  @Test func `requests send diagnostics, and the chain crosses sessions`() async throws {
    let diagnostics = ClaudeCacheDiagnostics()
    let transport = MockTransport(body: textTurn(deltas: ["Done!"]))
    let model = StubbedClaudeModel(transport: transport, cacheDiagnostics: diagnostics)
    _ = try await LanguageModelSession(model: model).respond(to: "one")
    _ = try await LanguageModelSession(model: model).respond(to: "two")

    let bodies = try transport.requests.map { request -> JSONValue in
      try JSONDecoder().decode(JSONValue.self, from: #require(request.httpBody))
    }
    #expect(bodies.count == 2)
    #expect(bodies.first?["diagnostics"] == ["previous_message_id": nil])
    #expect(bodies.last?["diagnostics"] == ["previous_message_id": "msg_1"])
  }

  @Test func `a model without diagnostics sends none`() async throws {
    let transport = MockTransport(body: textTurn(deltas: ["Done!"]))
    _ = try await LanguageModelSession(model: StubbedClaudeModel(transport: transport))
      .respond(to: "one")
    let body = try JSONDecoder().decode(
      JSONValue.self,
      from: #require(transport.lastRequest?.httpBody)
    )
    #expect(body["diagnostics"] == nil)
  }
}
