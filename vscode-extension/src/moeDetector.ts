import * as vscode from 'vscode';

interface MoEApiCall {
    line: number;
    provider: string;
    endpoint: string;
    potentialSavings: string;
}

export class MoEDetector {
    private patterns: { regex: RegExp; provider: string; endpoint: string; savings: string }[] = [
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

    analyze(document: vscode.TextDocument): MoEApiCall[] {
        const text = document.getText();
        const lines = text.split('\n');
        const calls: MoEApiCall[] = [];

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
    createDiagnostics(document: vscode.TextDocument, calls: MoEApiCall[]): vscode.Diagnostic[] {
        return calls.map(call => {
            const line = document.lineAt(call.line - 1);
            const range = new vscode.Range(
                call.line - 1, 0,
                call.line - 1, line.text.length
            );

            const diagnostic = new vscode.Diagnostic(
                range,
                `${call.provider} API call detected. Adaptive-K could save ${call.potentialSavings}`,
                vscode.DiagnosticSeverity.Information
            );

            diagnostic.source = 'Adaptive-K';
            diagnostic.code = 'moe-optimization';

            return diagnostic;
        });
    }

    /**
     * Get optimization suggestions for detected calls
     */
    getSuggestions(calls: MoEApiCall[]): string[] {
        const suggestions: string[] = [];

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
