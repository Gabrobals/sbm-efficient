import * as vscode from 'vscode';

export class DashboardPanel {
    public static currentPanel: DashboardPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];

    public static createOrShow(extensionUri: vscode.Uri): void {
        const column = vscode.ViewColumn.Beside;

        if (DashboardPanel.currentPanel) {
            DashboardPanel.currentPanel._panel.reveal(column);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'adaptiveKDashboard',
            'Adaptive-K Dashboard',
            column,
            {
                enableScripts: true,
                localResourceRoots: [extensionUri],
                retainContextWhenHidden: true
            }
        );

        DashboardPanel.currentPanel = new DashboardPanel(panel, extensionUri);
    }

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._extensionUri = extensionUri;

        this._update();

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        this._panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'calculate':
                        this._handleCalculation(message);
                        break;
                    case 'openDocs':
                        vscode.env.openExternal(vscode.Uri.parse('https://adaptive-k.vertexdata.it'));
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    private _handleCalculation(message: any): void {
        // Handle ROI calculations from webview
        const { tokens, model, savingsRate } = message;
        vscode.window.showInformationMessage(
            `Calculated savings: ${savingsRate}% for ${tokens} tokens using ${model}`
        );
    }

    private _update(): void {
        this._panel.webview.html = this._getHtmlContent();
    }

    private _getHtmlContent(): string {
        const config = vscode.workspace.getConfiguration('adaptive-k');
        const defaultModel = config.get<string>('defaultModel', 'deepseek-v3');
        const savingsRate = config.get<number>('savingsRate', 35);

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptive-K Dashboard</title>
    <style>
        :root {
            --vs-bg: #1e1e1e;
            --vs-surface: #252526;
            --vs-border: #3c3c3c;
            --vs-text: #cccccc;
            --vs-muted: #808080;
            --vs-blue: #3794ff;
            --vs-green: #4ec9b0;
            --vs-yellow: #dcdcaa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--vscode-font-family, 'Segoe UI', sans-serif);
            background: var(--vscode-editor-background, var(--vs-bg));
            color: var(--vscode-editor-foreground, var(--vs-text));
            padding: 20px;
            line-height: 1.6;
        }
        h1 { 
            color: var(--vs-green); 
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        h2 {
            color: var(--vscode-editor-foreground);
            font-size: 1.1rem;
            margin: 1.5rem 0 0.75rem;
            border-bottom: 1px solid var(--vs-border);
            padding-bottom: 0.5rem;
        }
        .card {
            background: var(--vscode-input-background, var(--vs-surface));
            border: 1px solid var(--vscode-input-border, var(--vs-border));
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
        .metric {
            text-align: center;
            padding: 1rem;
            background: var(--vs-bg);
            border-radius: 4px;
        }
        .metric .value {
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--vs-green);
        }
        .metric .label {
            color: var(--vs-muted);
            font-size: 0.85rem;
        }
        .metric.blue .value { color: var(--vs-blue); }
        label {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--vs-muted);
            font-size: 0.9rem;
        }
        select, input {
            width: 100%;
            padding: 0.5rem;
            background: var(--vscode-input-background);
            border: 1px solid var(--vscode-input-border, var(--vs-border));
            color: var(--vscode-input-foreground, var(--vs-text));
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        button {
            background: var(--vs-blue);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        button:hover { opacity: 0.9; }
        .result {
            background: rgba(78, 201, 176, 0.1);
            border: 1px solid rgba(78, 201, 176, 0.3);
            padding: 1rem;
            border-radius: 6px;
            margin-top: 1rem;
        }
        .result-row {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--vs-border);
        }
        .result-row:last-child { border-bottom: none; }
        .result-row .savings { color: var(--vs-green); font-weight: bold; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        th, td {
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid var(--vs-border);
        }
        th { color: var(--vs-muted); }
        .status { color: var(--vs-green); }
        a { color: var(--vs-blue); text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🚀 Adaptive-K Dashboard</h1>
    
    <div class="grid">
        <div class="metric">
            <div class="value">30.4%</div>
            <div class="label">Avg Compute Savings</div>
        </div>
        <div class="metric blue">
            <div class="value">33.3%</div>
            <div class="label">Max Savings (Nemotron 3)</div>
        </div>
    </div>

    <h2>💰 ROI Calculator</h2>
    <div class="card">
        <label>Daily Token Volume</label>
        <select id="tokenVolume">
            <option value="1000000">1M tokens/day</option>
            <option value="10000000">10M tokens/day</option>
            <option value="100000000" selected>100M tokens/day</option>
            <option value="1000000000">1B tokens/day</option>
        </select>
        
        <label>MoE Model</label>
        <select id="model">
            <option value="0.14" ${defaultModel === 'deepseek-v3' ? 'selected' : ''}>DeepSeek-V3 ($0.14/1M)</option>
            <option value="0.24" ${defaultModel === 'mixtral-8x7b' ? 'selected' : ''}>Mixtral 8x7B ($0.24/1M)</option>
            <option value="0.65" ${defaultModel === 'qwen-moe' ? 'selected' : ''}>Qwen-MoE ($0.65/1M)</option>
        </select>
        
        <label>Savings Rate: <span id="rateDisplay">${savingsRate}%</span></label>
        <input type="range" id="savingsRate" min="20" max="55" value="${savingsRate}" />
        
        <button onclick="calculate()">Calculate Savings</button>
        
        <div class="result" id="result" style="display: none;">
            <div class="result-row">
                <span>Monthly Baseline</span>
                <span id="baseline">$0</span>
            </div>
            <div class="result-row">
                <span>With Adaptive-K</span>
                <span class="savings" id="adaptive">$0</span>
            </div>
            <div class="result-row">
                <span>Monthly Savings</span>
                <span class="savings" id="monthly">$0</span>
            </div>
            <div class="result-row">
                <span>Annual Savings</span>
                <span class="savings" id="annual">$0</span>
            </div>
        </div>
    </div>

    <h2>📊 Validated Models</h2>
    <div class="card">
        <table>
            <tr>
                <th>Model</th>
                <th>Experts</th>
                <th>Savings</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Mixtral 8x7B</td>
                <td>8</td>
                <td>31.0%</td>
                <td class="status">✓ Validated</td>
            </tr>
            <tr>
                <td>Qwen1.5-MoE</td>
                <td>60</td>
                <td>32.4%</td>
                <td class="status">✓ Validated</td>
            </tr>
            <tr>
                <td>OLMoE 1B-7B</td>
                <td>64</td>
                <td>24.7%</td>
                <td class="status">✓ Validated</td>
            </tr>
        </table>
    </div>

    <h2>🔗 Resources</h2>
    <div class="card">
        <p><a href="#" onclick="openDocs()">📖 Documentation</a></p>
        <p><a href="#" onclick="openPyPI()">📦 PyPI Package</a></p>
        <p><a href="#" onclick="openGitHub()">💻 GitHub Repository</a></p>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        
        document.getElementById('savingsRate').addEventListener('input', function() {
            document.getElementById('rateDisplay').textContent = this.value + '%';
        });
        
        function calculate() {
            const tokens = parseInt(document.getElementById('tokenVolume').value);
            const price = parseFloat(document.getElementById('model').value);
            const rate = parseInt(document.getElementById('savingsRate').value);
            
            const dailyCost = (tokens / 1000000) * price;
            const monthlyCost = dailyCost * 30;
            const adaptiveCost = monthlyCost * (1 - rate/100);
            const savings = monthlyCost - adaptiveCost;
            
            document.getElementById('baseline').textContent = '$' + monthlyCost.toLocaleString(undefined, {maximumFractionDigits: 0});
            document.getElementById('adaptive').textContent = '$' + adaptiveCost.toLocaleString(undefined, {maximumFractionDigits: 0});
            document.getElementById('monthly').textContent = '$' + savings.toLocaleString(undefined, {maximumFractionDigits: 0});
            document.getElementById('annual').textContent = '$' + (savings * 12).toLocaleString(undefined, {maximumFractionDigits: 0});
            document.getElementById('result').style.display = 'block';
            
            vscode.postMessage({ command: 'calculate', tokens, model: price, savingsRate: rate });
        }
        
        function openDocs() {
            vscode.postMessage({ command: 'openDocs' });
        }
        
        function openPyPI() {
            vscode.postMessage({ command: 'openUrl', url: 'https://pypi.org/project/adaptive-k-routing/' });
        }
        
        function openGitHub() {
            vscode.postMessage({ command: 'openUrl', url: 'https://github.com/Gabrobals/sbm-efficient' });
        }
        
        // Auto-calculate on load
        calculate();
    </script>
</body>
</html>`;
    }

    public dispose(): void {
        DashboardPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const d = this._disposables.pop();
            if (d) {
                d.dispose();
            }
        }
    }
}
