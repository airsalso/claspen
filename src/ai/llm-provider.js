// Copyright (C) 2025 Keygraph, Inc.
// Generic LLM Provider Interface

export class LLMProvider {
  /**
   * Run a prompt and return a stream of turns/messages.
   * @param {Object} params
   * @param {string} params.prompt
   * @param {Object} params.options
   * @returns {AsyncIterable}
   */
  async *query({ prompt, options }) {
    throw new Error('Method query() must be implemented');
  }
}
