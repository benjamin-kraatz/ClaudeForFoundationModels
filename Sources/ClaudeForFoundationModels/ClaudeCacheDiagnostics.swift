// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import ClaudeAPI
import Foundation
import Synchronization

/// Opts a model's requests in to the API's cache diagnostics, and collects what
/// the API reports.
///
/// Pass one instance to ``ClaudeLanguageModel/init(name:auth:userProfileID:fixedEffort:fallbacks:cacheDiagnostics:serverTools:baseURL:timeout:)``
/// and keep it for as long as you want requests compared. Each request names
/// the response before it as `diagnostics.previous_message_id`, so the chain
/// follows this instance, not a session: requests of a tool loop compare
/// against each other, and the first request of a fresh session compares
/// against the last request of the one before. Share an instance only between
/// requests that run one after another, because the API compares each request
/// to the previous one sent.
///
/// Diagnostics are best-effort: they never block or fail a request. They work
/// on the Claude API only, so don't use them with a `baseURL` that reaches
/// another platform. The API keeps a short-lived fingerprint of hashes and
/// token estimates for each opted-in request, never the prompt itself.
public final class ClaudeCacheDiagnostics: Sendable {
  /// What the API said about one response, read from its `message_start`.
  public struct Report: Sendable, Hashable {
    public enum Outcome: Sendable, Hashable {
      /// The instance had no earlier response to compare this one against.
      case firstRequest
      /// The request matched the previous one up to where it was appended to.
      /// A miss here means the cache entry was gone, not that the prompt changed.
      case noDivergence
      /// The comparison was still running when the response started.
      case pending
      /// The API's `cache_miss_reason`. `reason` is its `type`, such as
      /// `system_changed`, kept as a string because the set can grow.
      /// `missedInputTokens` is the API's estimate of the prompt after the
      /// divergence, a magnitude rather than a billing number, and is `nil`
      /// for `previous_message_not_found` and `unavailable`.
      case cacheMiss(reason: String, missedInputTokens: Int?)
    }

    public let messageID: String
    /// The response this one was compared against.
    public let previousMessageID: String?
    public let outcome: Outcome
    /// The uncached remainder of the prompt.
    public let inputTokens: Int
    public let cacheReadInputTokens: Int
    public let cacheCreationInputTokens: Int
  }

  private struct State {
    var previousMessageID: String?
    var reports: [Report] = []
  }

  private let state = Mutex(State())

  public init() {}

  /// The reports collected since the last call, oldest first, and forgets them.
  /// The chain of previous responses carries on.
  public func takeReports() -> [Report] {
    state.withLock { state in
      defer { state.reports = [] }
      return state.reports
    }
  }

  /// The `diagnostics` object for the next request.
  var nextRequest: CacheDiagnosticsRequest {
    state.withLock { .init(previousMessageID: $0.previousMessageID) }
  }

  /// Files a response's `message_start`, which carries the diagnosis and the
  /// prompt-side usage, and makes it the response the next request names.
  func record(_ response: MessagesResponse) {
    state.withLock { state in
      state.reports.append(
        Report(
          messageID: response.id,
          previousMessageID: state.previousMessageID,
          outcome: .init(diagnostics: response.diagnostics, comparedTo: state.previousMessageID),
          inputTokens: response.usage.inputTokens ?? 0,
          cacheReadInputTokens: response.usage.cacheReadInputTokens ?? 0,
          cacheCreationInputTokens: response.usage.cacheCreationInputTokens ?? 0
        )
      )
      state.previousMessageID = response.id
    }
  }
}

extension ClaudeCacheDiagnostics: Hashable {
  /// Identity: two instances are two chains, and ``ClaudeExecutor/Configuration``
  /// is keyed on which chain it feeds.
  public static func == (lhs: ClaudeCacheDiagnostics, rhs: ClaudeCacheDiagnostics) -> Bool {
    lhs === rhs
  }

  public func hash(into hasher: inout Hasher) {
    hasher.combine(ObjectIdentifier(self))
  }
}

extension ClaudeCacheDiagnostics.Report.Outcome {
  /// Reads the response's `diagnostics` field. The API answers `null` both for
  /// a first turn and for a comparison that found nothing, so `previous` tells
  /// those apart. Any object without a usable reason is inconclusive, like the
  /// documented `{"cache_miss_reason": null}`.
  init(diagnostics: JSONValue?, comparedTo previous: String?) {
    guard previous != nil else {
      self = .firstRequest
      return
    }
    guard let diagnostics else {
      self = .noDivergence
      return
    }
    guard case .string(let reason)? = diagnostics["cache_miss_reason"]?["type"] else {
      self = .pending
      return
    }
    var missedInputTokens: Int?
    if case .number(let tokens)? = diagnostics["cache_miss_reason"]?["cache_missed_input_tokens"] {
      missedInputTokens = Int(exactly: tokens)
    }
    self = .cacheMiss(reason: reason, missedInputTokens: missedInputTokens)
  }
}
