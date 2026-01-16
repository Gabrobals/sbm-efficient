"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.MoEDetector = void 0;
const vscode = __importStar(require("vscode"));
class MoEDetector {
    patterns = [
        // OpenAI / Azure OpenAI
        {
            regex: /openai\.chat\.completions\.create|client\.chat\.completions\.create/gi,
            provider: 'OpenAI',
            endpoint: 'chat.completions',
            savings: '20-35%'
        },
        // Together.ai
        {
            regex: /together\.complete|together\.chat|TogetherClient/gi,
            provider: 'Together.ai',
            endpoint: 'inference',
            savings: '35-52%'
        },
        // DeepSeek
        {
            regex: /deepseek|DeepSeekClient/gi,
            provider: 'DeepSeek',
            endpoint: 'chat',
            savings: '30-40%'
        },
        // Mixtral specific
        {
            regex: /mixtral|mistral.*moe/gi,
            provider: 'Mistral MoE',
            endpoint: 'inference',
            savings: '45-52%'
        },
        // Generic MoE patterns
        {
            regex: /mixture.of.experts|moe_model|expert_routing/gi,
            provider: 'MoE Framework',
            endpoint: 'custom',
            savings: '25-50%'
        },
        // Ollama (local MoE)
        {
            regex: /ollama\.generate|ollama\.chat/gi,
            provider: 'Ollama',
            endpoint: 'local',
            savings: '30-40%'
        },
        // vLLM
        {
            regex: /vllm|LLM\(.*model=/gi,
            provider: 'vLLM',
            endpoint: 'inference',
            savings: '35-50%'
        },
        // HuggingFace
        {
            regex: /transformers.*AutoModelForCausalLM|pipeline\(.*text-generation/gi,
            provider: 'HuggingFace',
            endpoint: 'transformers',
            savings: '25-45%'
        }
    ];
    analyze(document) {
        const text = document.getText();
        const lines = text.split('\n');
        const calls = [];
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            for (const pattern of this.patterns) {
                if (pattern.regex.test(line)) {
                    calls.push({
                        line: i + 1,
                        provider: pattern.provider,
                        endpoint: pattern.endpoint,
                        potentialSavings: pattern.savings
                    });
                    // Reset regex lastIndex
                    pattern.regex.lastIndex = 0;
                    break; // One match per line
                }
            }
        }
        return calls;
    }
    /**
     * Create diagnostic collection for highlighting MoE calls
     */
    createDiagnostics(document, calls) {
        return calls.map(call => {
            const line = document.lineAt(call.line - 1);
            const range = new vscode.Range(call.line - 1, 0, call.line - 1, line.text.length);
            const diagnostic = new vscode.Diagnostic(range, `${call.provider} API call detected. Adaptive-K could save ${call.potentialSavings}`, vscode.DiagnosticSeverity.Information);
            diagnostic.source = 'Adaptive-K';
            diagnostic.code = 'moe-optimization';
            return diagnostic;
        });
    }
    /**
     * Get optimization suggestions for detected calls
     */
    getSuggestions(calls) {
        const suggestions = [];
        const providers = [...new Set(calls.map(c => c.provider))];
        if (providers.includes('OpenAI')) {
            suggestions.push('Consider using DeepSeek-V3 or Mixtral for MoE optimization');
        }
        if (providers.includes('Together.ai')) {
            suggestions.push('Together.ai already supports MoE models - enable Adaptive-K routing');
        }
        if (providers.includes('DeepSeek')) {
            suggestions.push('DeepSeek-V3 uses 256 experts - Adaptive-K can reduce to ~100 active');
        }
        if (calls.length > 5) {
            suggestions.push('High API usage detected. Adaptive-K could provide significant savings.');
        }
        return suggestions;
    }
}
exports.MoEDetector = MoEDetector;
//# sourceMappingURL=moeDetector.js.map