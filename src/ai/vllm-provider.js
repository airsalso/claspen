// Copyright (C) 2025 Keygraph, Inc.
// vLLM (OpenAI-compatible) Provider Implementation with Full Toolset Aliases

import OpenAI from 'openai';
import { spawn } from 'child_process';
import { fs, path, $ } from 'zx';
import { LLMProvider } from './llm-provider.js';

const MAX_OUTPUT_CHARS = parseInt(process.env.VLLM_MAX_OUTPUT_CHARS) || 4000;
const MAX_HISTORY_TURNS = parseInt(process.env.VLLM_MAX_HISTORY_TURNS) || 10;

class StdioMcpClient {
  constructor(name, command, args, env) {
    this.name = name; this.command = command; this.args = args; this.env = env;
    this.process = null; this.requestId = 1; this.pendingRequests = new Map(); this.buffer = '';
  }
  async start() {
    this.process = spawn(this.command, this.args, { env: { ...process.env, ...this.env }, stdio: ['pipe', 'pipe', 'inherit'] });
    this.process.stdout.on('data', (d) => { this.buffer += d.toString(); this.pB(); });
    await this.request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 's-vllm', version: '1.0' } });
    this.sendNotification('notifications/initialized');
  }
  pB() {
    let nl; while ((nl = this.buffer.indexOf('\n')) !== -1) {
      const l = this.buffer.slice(0, nl); this.buffer = this.buffer.slice(nl + 1);
      try { const r = JSON.parse(l); if (r.id && this.pendingRequests.has(r.id)) { const { resolve, reject } = this.pendingRequests.get(r.id); this.pendingRequests.delete(r.id); if (r.error) reject(new Error(r.error.message)); else resolve(r.result); } } catch (e) {}
    }
  }
  request(m, p) { return new Promise((resolve, reject) => { const id = this.requestId++; this.pendingRequests.set(id, { resolve, reject }); this.process.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method: m, params: p }) + '\n'); }); }
  sendNotification(m, p) { this.process.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: m, params: p }) + '\n'); }
  async listTools() { const r = await this.request('tools/list', {}); return r.tools || []; }
  async callTool(n, a) { return await this.request('tools/call', { name: n, arguments: a }); }
  stop() { if (this.process) { this.process.kill(); this.process = null; } }
}

export class VLLMProvider extends LLMProvider {
  constructor() {
    super();
    this.client = new OpenAI({ baseURL: process.env.VLLM_BASE_URL || 'http://localhost:8000/v1', apiKey: process.env.VLLM_API_KEY || 'no-key-needed' });
    this.model = process.env.VLLM_MODEL || 'openai/gpt-oss-20b';
    $.verbose = false;
  }
  async *query({ prompt, options }) {
    const { mcpServers = {}, maxTurns = 10000, cwd, agentName } = options;
    this.currentAgentName = agentName;
    const startTime = Date.now(); let turnCount = 0; const activeClients = new Map(); const toolMap = new Map();
    const lastToolCalls = new Map();
    const savedDeliverableLengths = new Map();
    try {
      const tools = []; this.addFsTools(tools, toolMap, cwd); this.addSdkTools(tools, toolMap, cwd);
      if (mcpServers['shannon-helper']) {
        for (const t of mcpServers['shannon-helper'].tools) {
          // Only add tool if it doesn't already exist to prioritize built-in SDK tools (like robust save_deliverable)
          if (!toolMap.has(t.name)) {
            tools.push({ type: 'function', function: { name: t.name, description: t.description, parameters: t.inputSchema } });
            toolMap.set(t.name, { originalName: t.name, isExternal: false, handler: t.handler });
          }
        }
      }
      for (const [name, cfg] of Object.entries(mcpServers)) {
        if (name === 'shannon-helper') continue;
        if (cfg.type === 'stdio') {
          try {
            const c = new StdioMcpClient(name, cfg.command, cfg.args, cfg.env); await c.start(); activeClients.set(name, c);
            const ts = await c.listTools(); for (const t of ts) { const fn = `${name}__${t.name}`; tools.push({ type: 'function', function: { name: fn, description: t.description, parameters: t.inputSchema } }); toolMap.set(fn, { client: c, originalName: t.name, isExternal: true }); }
          } catch (e) {}
        }
      }
      yield { type: 'system', subtype: 'init', model: this.model, permissionMode: 'bypassPermissions', mcp_servers: Object.keys(mcpServers).map(n => ({ name: n, status: 'connected' })) };
      let messages = [ { role: 'system', content: prompt } ];
      while (turnCount < maxTurns) {
        turnCount++;
        let activeMsgs = messages.length > MAX_HISTORY_TURNS * 2 + 1 ? [ messages[0], { role: 'system', content: `[Context history truncated at turn ${turnCount}. If you have already saved your deliverable and announced completion, simply say "Done." to end the session.]` }, ...messages.slice(-(MAX_HISTORY_TURNS * 2)) ] : messages;
        if (turnCount > (parseInt(process.env.VLLM_TURN_WARNING_THRESHOLD) || 30)) {
          const maxTurnsAgent = parseInt(process.env.SHANNON_AGENT_MAX_TURNS) || 120;
          activeMsgs = [...activeMsgs, { role: 'system', content: `[WARNING: Turn ${turnCount}/${maxTurnsAgent}. You MUST finalize your analysis and call save_deliverable NOW. Do NOT start new Task calls. If you are finished, say "Done."]` }];
        }
        const res = await (async () => {
          try {
            return await this.client.chat.completions.create({
              model: this.model,
              messages: activeMsgs,
              tools: tools.length > 0 ? tools : undefined,
              max_tokens: parseInt(process.env.VLLM_MAX_TOKENS) || 3000
            });
          } catch (e) {
            if (e.message.includes('tokens') || e.message.includes('context')) {
              // Try one more time with super aggressive truncation
              const minimalMsgs = [ activeMsgs[0], { role: 'system', content: '[CRITICAL: Context limit reached. Providing minimal history to avoid crash.]' }, ...activeMsgs.slice(-2) ];
              return await this.client.chat.completions.create({
                model: this.model,
                messages: minimalMsgs,
                tools: tools.length > 0 ? tools : undefined,
                max_tokens: parseInt(process.env.VLLM_RETRY_MAX_TOKENS) || 1000
              });
            }
            throw e;
          }
        })();

        const asstMsg = res.choices[0].message;
        if (!asstMsg.tool_calls && asstMsg.content && (asstMsg.content.includes('{') || asstMsg.content.includes('save_deliverable'))) {
           const jsonMatch = asstMsg.content.match(/```(?:json)?\s*({[\s\S]*?})\s*```/);
           if (jsonMatch) {
             try { const parsed = JSON.parse(jsonMatch[1]); const toolName = parsed.tool || parsed.name || (parsed.deliverable_type ? 'save_deliverable' : null);
               if (toolName && (toolMap.has(toolName) || ['search', 'read', 'grep', 'cat', 'TaskAgent'].includes(toolName))) asstMsg.tool_calls = [{ id: 'manual-' + turnCount, type: 'function', function: { name: toolName, arguments: jsonMatch[1] } }];
             } catch(e) {}
           }
        }
        messages.push(asstMsg); yield { type: 'assistant', message: { content: asstMsg.content || '' } };
        if (asstMsg.tool_calls) {
          for (const tc of asstMsg.tool_calls) {
             let tn = tc.function.name.split(/[ <|]/)[0];
             if (tn === 'search' || tn === 'search_file') tn = 'grep';
             if (tn === 'read' || tn === 'open_file' || tn === 'readFile') tn = 'cat';
             if (tn === 'TaskAgent') tn = 'Task';
             let tIn; try { tIn = JSON.parse(tc.function.arguments); } catch (e) { continue; }
             const normTask = tn === 'Task' ? JSON.stringify(tIn).toLowerCase().replace(/[^a-z0-9]/g, '') : JSON.stringify(tIn);
             const callKey = tn === 'Task' ? `Task:${normTask.slice(0, 200)}` : `${tn}:${JSON.stringify(tIn).slice(0, 100)}`;
             const count = (lastToolCalls.get(callKey) || 0) + 1; lastToolCalls.set(callKey, count);
            const totalTaskCount = (lastToolCalls.get('TOTAL_TASKS') || 0) + (tn === 'Task' ? 1 : 0);
            if (tn === 'Task') lastToolCalls.set('TOTAL_TASKS', totalTaskCount);

            yield { type: 'tool_use', name: tn, input: tIn, id: tc.id };
            let tRes;
            if (count > (parseInt(process.env.VLLM_MAX_REPEATED_TOOL_CALLS) || 2) && !['save_deliverable', 'bash', 'cat', 'grep', 'TodoWrite'].includes(tn)) {
              tRes = "SYSTEM: Repeated tool call. You have already tried this task. Use cat/grep directly or finalize your report.";
            } else if (totalTaskCount > (parseInt(process.env.VLLM_MAX_TASK_DELEGATIONS) || 10) && tn === 'Task') {
              tRes = "SYSTEM: Maximum Task delegations reached for this session. You must now synthezise existing information using cat/grep and save your report.";
            } else {
              const tInf = toolMap.get(tn);
              if (tInf) {
                try {
                  if (tInf.isExternal) { const r = await tInf.client.callTool(tInf.originalName, tIn); const txt = r.content.find(c => c.type === 'text'); tRes = txt ? txt.text : r.content; }
                  else {
                    // Safety check for multiple save_deliverable calls (Regression Protection)
                    if (tn === 'save_deliverable') {
                      const type = (tIn.deliverable_type || tIn.type || '').toUpperCase().replace(/-/g, '_');
                      const newContent = tIn.content || '';
                      const prevLength = savedDeliverableLengths.get(type);

                      // If we already saved this type, check for major regression (shortening)
                      if (prevLength !== undefined && !tIn.force) {
                        const regressionThreshold = 0.5; // Block if new content is < 50% of previous
                        if (newContent.length < prevLength * regressionThreshold) {
                          tRes = `SYSTEM: WARNING! Your new report is significantly shorter (${newContent.length} bytes) than the previously saved version (${prevLength} bytes). This usually happens when you summarize away technical details. Overwrite BLOCKED to prevent data loss. If you MUST save this shorter version, explain why or add more detail. If you are finished, just say "Done."`;
                        } else {
                          tRes = await tInf.handler(tIn, { provider: this });
                          if (tRes && tRes.status === 'success') savedDeliverableLengths.set(tRes.deliverableType || type, newContent.length);
                        }
                      } else {
                        tRes = await tInf.handler(tIn, { provider: this });
                        if (tRes && tRes.status === 'success') savedDeliverableLengths.set(tRes.deliverableType || type, newContent.length);
                      }
                    } else {
                      tRes = await tInf.handler(tIn, { provider: this });
                    }
                  }
                } catch (e) { tRes = { status: 'error', message: e.message }; }
              } else tRes = { status: 'error', message: `Tool ${tn} not found` };
            }
            if (typeof tRes === 'string' && tRes.length > MAX_OUTPUT_CHARS) tRes = tRes.slice(0, MAX_OUTPUT_CHARS) + `\n\n[Truncated]`;
            yield { type: 'tool_result', name: tn, content: tRes, id: tc.id };
            messages.push({ role: 'tool', tool_call_id: tc.id, name: tn, content: typeof tRes === 'string' ? tRes : JSON.stringify(tRes) });
          }
        } else {
          const content = (asstMsg.content || "").toLowerCase();
          const isStop = res.choices[0].finish_reason === 'stop';
          const hasDone = content.includes('done') || content.includes('complete') || content.includes('finished') || content.includes('saved');
          if (isStop || (content.length < (parseInt(process.env.VLLM_DONE_DETECTION_LENGTH) || 100) && hasDone)) {
            yield { type: 'result', result: asstMsg.content || "Done.", duration_ms: Date.now() - startTime, total_cost_usd: 0 };
            return;
          }
        }
      }
    } finally { for (const c of activeClients.values()) c.stop(); }
  }
  addFsTools(ts, tm, bd) {
    const list = [
      { name: 'ls', desc: 'List files', params: { type: 'object', properties: { path: { type: 'string' } } }, h: async (a) => {
          const pStr = a.path || '.';
          const target = path.isAbsolute(pStr) ? pStr : path.join(bd, pStr);
          const files = (await fs.readdir(target)).sort();
          if (files.length > 50) return `Files in ${pStr} (${files.length} total):\n` + files.slice(0, 25).join('\n') + '\n...\n' + files.slice(-25).join('\n');
          return files.join('\n') || "(Empty)";
      }},
      { name: 'cat', desc: 'Read file', params: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] }, h: async (a) => {
          if (!a.path) throw new Error("path is required");
          const target = path.isAbsolute(a.path) ? a.path : path.join(bd, a.path);
          return await fs.readFile(target, 'utf8');
      }},
      { name: 'grep', desc: 'Search pattern', params: { type: 'object', properties: { pattern: { type: 'string' }, query: { type: 'string' }, path: { type: 'string' } } }, h: async (a) => {
          const pStr = a.path || '.';
          return (await $({ cwd: bd })`grep -rnI ${a.pattern || a.query} ${pStr}`.nothrow()).stdout;
      }},
      { name: 'bash', desc: 'Run shell command', params: { type: 'object', properties: { cmd: { type: 'string' } }, required: ['cmd'] }, h: async (a) => {
          const cmdStr = Array.isArray(a.cmd) ? a.cmd.join(' ') : a.cmd;
          const res = await $({ cwd: bd })`bash -c ${cmdStr}`.nothrow();
          return res.stdout + res.stderr;
      }},
    ];
    for (const l of list) {
      ts.push({ type: 'function', function: { name: l.name, description: l.desc, parameters: l.params } });
      tm.set(l.name, { isExternal: false, handler: l.h, parameters: l.params });
    }
    // Explicitly add aliases
    const grepTool = tm.get('grep');
    if (grepTool) {
      tm.set('search', { isExternal: false, handler: grepTool.handler, parameters: grepTool.parameters });
      ts.push({ type: 'function', function: { name: 'search', description: 'Alias for grep', parameters: grepTool.parameters } });
    }
    const catTool = tm.get('cat');
    if (catTool) {
      tm.set('read', { isExternal: false, handler: catTool.handler, parameters: catTool.parameters });
      ts.push({ type: 'function', function: { name: 'read', description: 'Alias for cat', parameters: catTool.parameters } });
      tm.set('open_file', { isExternal: false, handler: catTool.handler, parameters: catTool.parameters });
      ts.push({ type: 'function', function: { name: 'open_file', description: 'Alias for cat', parameters: catTool.parameters } });
    }
    const bashTool = tm.get('bash');
    if (bashTool) {
      tm.set('execute_command', { isExternal: false, handler: bashTool.handler, parameters: bashTool.parameters });
      ts.push({ type: 'function', function: { name: 'execute_command', description: 'Alias for bash', parameters: bashTool.parameters } });
    }
  }
  addSdkTools(ts, tm, bd) {
    const list = [
      { name: 'TodoWrite', desc: 'Update tasks', params: { type: 'object', properties: { tasks: { type: 'array' } } }, h: (a) => 'Updated' },
      { name: 'Task', desc: 'Delegate task', params: { type: 'object', properties: { task: { type: 'string' } }, required: ['task'] }, h: async (a, { provider }) => {
          const subTools = []; const subToolMap = new Map(); provider.addFsTools(subTools, subToolMap, bd);
          let subMsgs = [{ role: 'system', content: `Analyze the project in ${bd}.\nTask: ${a.task}.\nYou MUST use ls/cat/grep to explore and provide specific technical findings in Markdown tables. Focus on actionable security intelligence. When you have enough information, provide a concise technical summary and STOP.` }];
          const subMaxSteps = parseInt(process.env.VLLM_SUBTASK_MAX_STEPS) || 40;
          const subHistoryDepth = parseInt(process.env.VLLM_SUBTASK_HISTORY_DEPTH) || 6;
          const subMaxTokens = parseInt(process.env.VLLM_SUBTASK_MAX_TOKENS) || 1000;

          const subToolResTruncLimit = parseInt(process.env.VLLM_SUBTASK_TOOL_RES_TRUNC_LIMIT) || 2500;
          for (let i = 0; i < subMaxSteps; i++) {
            // Aggressive truncation for sub-tasks (keep system prompt + last N messages)
            let activeSubMsgs = subMsgs.length > (subHistoryDepth + 2) ? [ subMsgs[0], { role: 'system', content: '[Sub-task history truncated to save context]' }, ...subMsgs.slice(-subHistoryDepth) ] : subMsgs;
            try {
              const res = await provider.client.chat.completions.create({
                model: provider.model,
                messages: activeSubMsgs,
                tools: subTools,
                max_tokens: subMaxTokens
              });
              const msg = res.choices[0].message; subMsgs.push(msg);
              if (msg.tool_calls) {
                for (const tc of msg.tool_calls) {
                  let tn = tc.function.name.split(/[ <|]/)[0];
                  if (tn === 'search' || tn === 'search_file') tn = 'grep';
                  if (tn === 'read' || tn === 'open_file' || tn === 'readFile') tn = 'cat';
                  let tIn;
                  try { tIn = JSON.parse(tc.function.arguments); } catch (e) { const m = tc.function.arguments.match(/\{[\s\S]*\}/); if (m) try { tIn = JSON.parse(m[0]); } catch(e2) {} }
                  if (!tIn) { subMsgs.push({ role: 'tool', tool_call_id: tc.id, name: tn, content: "Error: Invalid JSON arguments." }); continue; }
                  const tInf = subToolMap.get(tn); let tRes;
                  if (tInf) { try { tRes = await tInf.handler(tIn, { provider }); } catch (e) { tRes = `Error: ${e.message}`; } } else tRes = `Tool ${tn} not found`;
                  if (typeof tRes === 'string' && tRes.length > subToolResTruncLimit) tRes = tRes.slice(0, subToolResTruncLimit) + '... [Truncated]';
                  subMsgs.push({ role: 'tool', tool_call_id: tc.id, name: tn, content: typeof tRes === 'string' ? tRes : JSON.stringify(tRes) });
                }
              } else return (msg.content || "Sub-task complete.").slice(0, MAX_OUTPUT_CHARS);
            } catch (e) {
              if (e.message.includes('token') || e.message.includes('context')) {
                const allFindings = subMsgs.map(m => {
                  if (m.role === 'assistant' && m.content) return m.content;
                  if (m.role === 'tool') return `(Tool ${m.name}): ${typeof m.content === 'object' ? JSON.stringify(m.content).slice(0, 300) : String(m.content).slice(0, 300).replace(/\n/g, ' ')}...`;
                  return null;
                }).filter(Boolean);

                const summary = allFindings.length > 20
                  ? [...allFindings.slice(0, 5), '... [Summary jump] ...', ...allFindings.slice(-15)].join('\n')
                  : allFindings.join('\n');

                return `Sub-task overflow recovery. Findings summary from partial history:\n${summary.slice(-5000) || "No findings available."}\n\n[SYSTEM: This task reached context limits. Do NOT retry the exact same task; use the findings above to synthesize your report or use cat/grep for specific details.]`;
              }
              throw e;
            }
          }
          const allFindings = subMsgs.map(m => {
            if (m.role === 'assistant' && m.content) return m.content;
            if (m.role === 'tool') return `(Tool ${m.name}): ${typeof m.content === 'object' ? JSON.stringify(m.content).slice(0, 300) : String(m.content).slice(0, 300).replace(/\n/g, ' ')}...`;
            return null;
          }).filter(Boolean);

          const summary = allFindings.length > 20
            ? [...allFindings.slice(0, 5), '... [Summary jump] ...', ...allFindings.slice(-15)].join('\n')
            : allFindings.join('\n');

          return `Sub-task execution reached turn limit. Partial technical findings gathered:\n${summary.slice(-5000) || "No findings available."}`;
      }},
      { name: 'save_deliverable', desc: 'Save report', params: { type: 'object', properties: { deliverable_type: { type: 'string' }, content: { type: 'string' } }, required: ['deliverable_type', 'content'] }, h: async (a) => await this.executeSaveDeliverable(a) }
    ];
    for (const l of list) { ts.push({ type: 'function', function: { name: l.name, description: l.desc, parameters: l.params } }); tm.set(l.name, { isExternal: false, handler: l.h }); }
  }
  async executeSaveDeliverable(i) {
    let t = i.deliverable_type || i.type;
    const { DeliverableType, DELIVERABLE_FILENAMES } = await import('../../mcp-server/src/types/deliverables.js');
    const { saveDeliverableFile } = await import('../../mcp-server/src/utils/file-operations.js');

    const u = String(t).toUpperCase().replace(/-/g, '_');
    const agent = this.currentAgentName || '';

    // Robust mapping for agent hallucinations
    if (DeliverableType[u]) {
      t = u;
    } else if (agent === 'pre-recon' || u.includes('CODE') || u.includes('ARCH')) {
      // Prioritize CODE_ANALYSIS for pre-recon, but allow others if they explicitly ask
      // If a vuln agent asks for CODE_ANALYSIS, maybe they really mean it, but usually it's a mistake.
      if (agent.includes('-vuln') || agent.includes('-exploit')) {
        // Redirect vuln agents to their specific analysis type if they use generic terms
        if (agent.includes('injection')) t = DeliverableType.INJECTION_ANALYSIS;
        else if (agent.includes('xss')) t = DeliverableType.XSS_ANALYSIS;
        else if (agent.includes('authz')) t = DeliverableType.AUTHZ_ANALYSIS;
        else if (agent.includes('auth')) t = DeliverableType.AUTH_ANALYSIS;
        else if (agent.includes('ssrf')) t = DeliverableType.SSRF_ANALYSIS;
        else t = DeliverableType.CODE_ANALYSIS;
      } else {
        t = DeliverableType.CODE_ANALYSIS;
      }
    } else if (u.includes('RECON') || agent === 'recon') {
      t = DeliverableType.RECON;
    } else if (u.includes('INJECTION') || agent.includes('injection')) {
      if (u.includes('QUEUE')) t = DeliverableType.INJECTION_QUEUE;
      else if (u.includes('EVIDENCE') || u.includes('EXPLOIT')) t = DeliverableType.INJECTION_EVIDENCE;
      else t = DeliverableType.INJECTION_ANALYSIS;
    } else if (u.includes('XSS')) {
      if (u.includes('QUEUE')) t = DeliverableType.XSS_QUEUE;
      else if (u.includes('EVIDENCE') || u.includes('EXPLOIT')) t = DeliverableType.XSS_EVIDENCE;
      else t = DeliverableType.XSS_ANALYSIS;
    } else if (u.includes('AUTHZ')) {
      if (u.includes('QUEUE')) t = DeliverableType.AUTHZ_QUEUE;
      else if (u.includes('EVIDENCE') || u.includes('EXPLOIT')) t = DeliverableType.AUTHZ_EVIDENCE;
      else t = DeliverableType.AUTHZ_ANALYSIS;
    } else if (u.includes('AUTH') || u.includes('SESSION')) {
      if (u.includes('QUEUE')) t = DeliverableType.AUTH_QUEUE;
      else if (u.includes('EVIDENCE') || u.includes('EXPLOIT')) t = DeliverableType.AUTH_EVIDENCE;
      else t = DeliverableType.AUTH_ANALYSIS;
    } else if (u.includes('SSRF')) {
      if (u.includes('QUEUE')) t = DeliverableType.SSRF_QUEUE;
      else if (u.includes('EVIDENCE') || u.includes('EXPLOIT')) t = DeliverableType.SSRF_EVIDENCE;
      else t = DeliverableType.SSRF_ANALYSIS;
    } else {
      t = DeliverableType.FINAL_REPORT;
    }

    const f = DELIVERABLE_FILENAMES[t];
    if (!f) throw new Error(`Invalid deliverable type: ${t}`);

    const p = saveDeliverableFile(f, i.content);
    return {
      status: 'success',
      message: `SUCCESS: Deliverable saved to ${f}. You MUST now conclude the session by saying "Done." or "PRE-RECON CODE ANALYSIS COMPLETE" depending on your instructions.`,
      filepath: p,
      deliverableType: t
    };
  }
}
