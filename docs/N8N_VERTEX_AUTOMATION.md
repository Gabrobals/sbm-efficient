# n8n Automation for VERTEX-RESEARCH Protocol

## Overview

This document describes how to automate the VERTEX-RESEARCH protocol using n8n workflows. Each phase has automated checkpoints, notifications, and quality gates.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VERTEX-RESEARCH n8n AUTOMATION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │ Phase 0 │───►│ Phase 1 │───►│ Phase 2 │───►│ Phase 3 │───►│ Phase 4 │   │
│  │ Problem │    │ LitRev  │    │ Theory  │    │ Hypoth  │    │ Design  │   │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘   │
│       │              │              │              │              │         │
│       ▼              ▼              ▼              ▼              ▼         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │  GATE   │    │  GATE   │    │  GATE   │    │  GATE   │    │  GATE   │   │
│  │ REVIEW  │    │ REVIEW  │    │ REVIEW  │    │ REVIEW  │    │ REVIEW  │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│                                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │Phase 10 │◄───│ Phase 9 │◄───│ Phase 8 │◄───│ Phase 7 │◄───│ Phase 6 │   │
│  │ Publish │    │ Paper   │    │ Analysis│    │ Valid   │    │ Exper   │   │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘   │
│       │              │              │              │              │         │
│       │              │              │              │              │         │
│       │              │              │         ┌────┴────┐         │         │
│       │              │              │         │ Phase 5 │◄────────┘         │
│       │              │              │         │  Impl   │                   │
│       │              │              │         └─────────┘                   │
│       │              │              │                                       │
│       ▼              ▼              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CENTRAL DATABASE                              │   │
│  │  (Notion / Airtable / PostgreSQL / Google Sheets)                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow 1: Project Initialization

**Trigger:** Manual or webhook when starting new research project

```json
{
  "name": "VERTEX-01-Project-Init",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "vertex-new-project",
        "method": "POST"
      }
    },
    {
      "name": "Create Project Structure",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const project = {\n  id: `PROJ-${Date.now()}`,\n  name: $input.item.json.project_name,\n  created: new Date().toISOString(),\n  current_phase: 0,\n  status: 'active',\n  hypotheses: [],\n  experiments: [],\n  gate_reviews: []\n};\nreturn { json: project };"
      }
    },
    {
      "name": "Create Notion Page",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "create",
        "databaseId": "{{ NOTION_DB_ID }}",
        "properties": {
          "Name": "={{ $json.name }}",
          "Phase": "Phase 0: Problem ID",
          "Status": "In Progress"
        }
      }
    },
    {
      "name": "Create GitHub Issue",
      "type": "n8n-nodes-base.github",
      "parameters": {
        "operation": "create",
        "owner": "Gabrobals",
        "repository": "sbm-efficient",
        "title": "Research: {{ $json.name }}",
        "body": "## New Research Project\\n\\nPhase: 0 - Problem Identification\\n\\n### Checklist\\n- [ ] Problem statement defined\\n- [ ] Significance established\\n- [ ] Scope bounded\\n- [ ] Success criteria specified"
      }
    },
    {
      "name": "Send Slack Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#research",
        "text": "🔬 New research project started: {{ $json.name }}"
      }
    }
  ]
}
```

---

## Workflow 2: Daily Research Standup

**Trigger:** Schedule (every morning at 9:00)

```json
{
  "name": "VERTEX-02-Daily-Standup",
  "nodes": [
    {
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": {
        "rule": {
          "interval": [{ "field": "cronExpression", "expression": "0 9 * * *" }]
        }
      }
    },
    {
      "name": "Get Active Projects",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "getAll",
        "databaseId": "{{ NOTION_DB_ID }}",
        "filters": {
          "property": "Status",
          "select": { "equals": "In Progress" }
        }
      }
    },
    {
      "name": "Generate Standup Report",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "message",
        "model": "gpt-4",
        "messages": [
          {
            "role": "system",
            "content": "You are a research assistant. Generate a brief daily standup summary."
          },
          {
            "role": "user", 
            "content": "Projects: {{ JSON.stringify($json) }}. Generate today's research priorities and blockers."
          }
        ]
      }
    },
    {
      "name": "Send Email Report",
      "type": "n8n-nodes-base.emailSend",
      "parameters": {
        "to": "gabriele.balsamo30@gmail.com",
        "subject": "🔬 VERTEX Daily Standup - {{ $now.format('yyyy-MM-dd') }}",
        "text": "={{ $json.choices[0].message.content }}"
      }
    }
  ]
}
```

---

## Workflow 3: Literature Review Automation

**Trigger:** When Phase 1 starts

```json
{
  "name": "VERTEX-03-Literature-Review",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "vertex-lit-review",
        "method": "POST"
      }
    },
    {
      "name": "Search Semantic Scholar",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "GET",
        "url": "https://api.semanticscholar.org/graph/v1/paper/search",
        "qs": {
          "query": "={{ $json.search_query }}",
          "limit": 50,
          "fields": "title,authors,year,abstract,citationCount,url"
        }
      }
    },
    {
      "name": "Search arXiv",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "GET",
        "url": "http://export.arxiv.org/api/query",
        "qs": {
          "search_query": "all:{{ $json.search_query }}",
          "max_results": 50
        }
      }
    },
    {
      "name": "Merge Results",
      "type": "n8n-nodes-base.merge",
      "parameters": {
        "mode": "append"
      }
    },
    {
      "name": "AI Relevance Scoring",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "message",
        "model": "gpt-4",
        "messages": [
          {
            "role": "system",
            "content": "Score each paper 1-10 for relevance to the research question. Return JSON array with paper_id and relevance_score."
          },
          {
            "role": "user",
            "content": "Research question: {{ $json.research_question }}\\n\\nPapers: {{ JSON.stringify($json.papers) }}"
          }
        ]
      }
    },
    {
      "name": "Filter Top Papers",
      "type": "n8n-nodes-base.filter",
      "parameters": {
        "conditions": {
          "number": [{ "value1": "={{ $json.relevance_score }}", "operation": "larger", "value2": 7 }]
        }
      }
    },
    {
      "name": "Create BibTeX Entry",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const papers = $input.all();\nlet bibtex = '';\nfor (const p of papers) {\n  bibtex += `@article{${p.json.id},\\n  title={${p.json.title}},\\n  author={${p.json.authors}},\\n  year={${p.json.year}}\\n}\\n\\n`;\n}\nreturn { json: { bibtex } };"
      }
    },
    {
      "name": "Save to Notion",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "create",
        "databaseId": "{{ NOTION_PAPERS_DB }}",
        "properties": {
          "Title": "={{ $json.title }}",
          "Relevance": "={{ $json.relevance_score }}",
          "Status": "To Review"
        }
      }
    }
  ]
}
```

---

## Workflow 4: Hypothesis Registration

**Trigger:** Manual form submission

```json
{
  "name": "VERTEX-04-Hypothesis-Registration",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "vertex-hypothesis",
        "method": "POST"
      }
    },
    {
      "name": "Validate Hypothesis",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "message",
        "model": "gpt-4",
        "messages": [
          {
            "role": "system",
            "content": "You are a Popperian research methodologist. Evaluate if the hypothesis is falsifiable. Check:\\n1. Is it falsifiable?\\n2. Are conditions clear?\\n3. Are success/failure criteria quantified?\\nReturn JSON: {valid: boolean, issues: string[], suggestions: string[]}"
          },
          {
            "role": "user",
            "content": "Hypothesis: {{ $json.hypothesis }}\\nConditions: {{ $json.conditions }}\\nFalsification criteria: {{ $json.falsification }}"
          }
        ]
      }
    },
    {
      "name": "Check Validity",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "boolean": [{ "value1": "={{ JSON.parse($json.choices[0].message.content).valid }}", "value2": true }]
        }
      }
    },
    {
      "name": "Register Valid Hypothesis",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "create",
        "databaseId": "{{ NOTION_HYPOTHESES_DB }}",
        "properties": {
          "ID": "={{ 'H' + ($now.toMillis() % 1000) }}",
          "Statement": "={{ $json.hypothesis }}",
          "Status": "Registered",
          "Registration Date": "={{ $now.toISO() }}"
        }
      }
    },
    {
      "name": "Git Commit Registration",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": {
        "command": "cd /path/to/project && echo '{{ $json.hypothesis }}' >> docs/REGISTERED_HYPOTHESES.md && git add . && git commit -m 'Register hypothesis {{ $json.id }}'"
      }
    },
    {
      "name": "Return Issues",
      "type": "n8n-nodes-base.respondToWebhook",
      "parameters": {
        "respondWith": "json",
        "responseBody": "={{ { status: 'invalid', issues: JSON.parse($json.choices[0].message.content).issues } }}"
      }
    }
  ]
}
```

---

## Workflow 5: Experiment Execution & Logging

**Trigger:** When experiment starts (webhook from Python script)

```json
{
  "name": "VERTEX-05-Experiment-Tracking",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "vertex-experiment",
        "method": "POST"
      }
    },
    {
      "name": "Route by Event",
      "type": "n8n-nodes-base.switch",
      "parameters": {
        "dataPropertyName": "event",
        "rules": [
          { "value": "start" },
          { "value": "checkpoint" },
          { "value": "complete" },
          { "value": "error" }
        ]
      }
    },
    {
      "name": "Log Start",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "create",
        "databaseId": "{{ NOTION_EXPERIMENTS_DB }}",
        "properties": {
          "ID": "={{ $json.experiment_id }}",
          "Status": "Running",
          "Start Time": "={{ $now.toISO() }}",
          "Config": "={{ JSON.stringify($json.config) }}",
          "Git Commit": "={{ $json.git_commit }}"
        }
      }
    },
    {
      "name": "Update Progress",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "update",
        "pageId": "={{ $json.notion_page_id }}",
        "properties": {
          "Progress": "={{ $json.progress }}%",
          "Current Seed": "={{ $json.current_seed }}"
        }
      }
    },
    {
      "name": "Log Completion",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "update",
        "pageId": "={{ $json.notion_page_id }}",
        "properties": {
          "Status": "Complete",
          "End Time": "={{ $now.toISO() }}",
          "Results": "={{ JSON.stringify($json.results) }}"
        }
      }
    },
    {
      "name": "Send Completion Alert",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#research",
        "text": "✅ Experiment {{ $json.experiment_id }} complete!\\n\\nResults:\\n{{ JSON.stringify($json.results, null, 2) }}"
      }
    },
    {
      "name": "Log Error",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "update",
        "pageId": "={{ $json.notion_page_id }}",
        "properties": {
          "Status": "Failed",
          "Error": "={{ $json.error_message }}"
        }
      }
    },
    {
      "name": "Send Error Alert",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#research",
        "text": "❌ Experiment {{ $json.experiment_id }} FAILED!\\n\\nError: {{ $json.error_message }}"
      }
    }
  ]
}
```

---

## Workflow 6: Gate Review Automation

**Trigger:** Manual or when phase deliverables are marked complete

```json
{
  "name": "VERTEX-06-Gate-Review",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "vertex-gate-review",
        "method": "POST"
      }
    },
    {
      "name": "Get Phase Checklist",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const checklists = {\n  0: ['Problem statement defined', 'Significance established', 'Scope bounded', 'Success criteria specified'],\n  1: ['Literature review complete', 'Gap analysis done', 'References collected'],\n  2: ['Mathematical framework defined', 'Assumptions documented', 'Notation consistent'],\n  3: ['Hypotheses falsifiable', 'Thresholds quantified', 'Pre-registered'],\n  4: ['Experimental matrix complete', 'Controls defined', 'Statistical plan ready'],\n  5: ['Code committed', 'Tests passing', 'Dependencies pinned'],\n  6: ['All seeds complete', 'Logs archived', 'No anomalies'],\n  7: ['Falsification checked', 'Statistical tests done', 'Limitations documented'],\n  8: ['Results interpreted', 'Visualizations created', 'Insights extracted'],\n  9: ['Paper draft complete', 'Figures high quality', 'References verified']\n};\nreturn { json: { checklist: checklists[$json.phase] } };"
      }
    },
    {
      "name": "AI Review",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "message",
        "model": "gpt-4",
        "messages": [
          {
            "role": "system",
            "content": "You are a research quality reviewer. Evaluate if all checklist items are satisfied based on the provided artifacts. Return JSON: {passed: boolean, missing: string[], recommendations: string[]}"
          },
          {
            "role": "user",
            "content": "Phase {{ $json.phase }} Gate Review\\n\\nChecklist: {{ JSON.stringify($json.checklist) }}\\n\\nArtifacts: {{ JSON.stringify($json.artifacts) }}"
          }
        ]
      }
    },
    {
      "name": "Check Pass/Fail",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "boolean": [{ "value1": "={{ JSON.parse($json.choices[0].message.content).passed }}", "value2": true }]
        }
      }
    },
    {
      "name": "Advance Phase",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "update",
        "pageId": "={{ $json.project_page_id }}",
        "properties": {
          "Phase": "={{ 'Phase ' + ($json.phase + 1) }}",
          "Last Gate": "={{ $now.toISO() }}"
        }
      }
    },
    {
      "name": "Send Success Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#research",
        "text": "✅ Gate Review PASSED for Phase {{ $json.phase }}!\\n\\nAdvancing to Phase {{ $json.phase + 1 }}"
      }
    },
    {
      "name": "Send Failure Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#research",
        "text": "⚠️ Gate Review FAILED for Phase {{ $json.phase }}\\n\\nMissing: {{ JSON.parse($json.choices[0].message.content).missing.join(', ') }}\\n\\nRecommendations: {{ JSON.parse($json.choices[0].message.content).recommendations.join(', ') }}"
      }
    }
  ]
}
```

---

## Workflow 7: Weekly Research Report

**Trigger:** Every Friday at 17:00

```json
{
  "name": "VERTEX-07-Weekly-Report",
  "nodes": [
    {
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": {
        "rule": {
          "interval": [{ "field": "cronExpression", "expression": "0 17 * * 5" }]
        }
      }
    },
    {
      "name": "Get All Projects",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "getAll",
        "databaseId": "{{ NOTION_DB_ID }}"
      }
    },
    {
      "name": "Get Experiments This Week",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "getAll",
        "databaseId": "{{ NOTION_EXPERIMENTS_DB }}",
        "filters": {
          "property": "Start Time",
          "date": { "past_week": {} }
        }
      }
    },
    {
      "name": "Generate Report",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "message",
        "model": "gpt-4",
        "messages": [
          {
            "role": "system",
            "content": "Generate a professional weekly research report in markdown format. Include: Executive Summary, Progress by Project, Key Results, Next Week Priorities, Blockers."
          },
          {
            "role": "user",
            "content": "Projects: {{ JSON.stringify($('Get All Projects').item.json) }}\\n\\nExperiments: {{ JSON.stringify($('Get Experiments This Week').item.json) }}"
          }
        ]
      }
    },
    {
      "name": "Save Report to Notion",
      "type": "n8n-nodes-base.notion",
      "parameters": {
        "operation": "create",
        "databaseId": "{{ NOTION_REPORTS_DB }}",
        "properties": {
          "Title": "Weekly Report - {{ $now.format('yyyy-MM-dd') }}",
          "Content": "={{ $json.choices[0].message.content }}"
        }
      }
    },
    {
      "name": "Send Email",
      "type": "n8n-nodes-base.emailSend",
      "parameters": {
        "to": "gabriele.balsamo30@gmail.com",
        "subject": "🔬 VERTEX Weekly Research Report - {{ $now.format('yyyy-MM-dd') }}",
        "html": "={{ $json.choices[0].message.content.replace(/\\n/g, '<br>') }}"
      }
    }
  ]
}
```

---

## Workflow 8: Pre-Publication Checklist

**Trigger:** When paper draft is marked complete

```json
{
  "name": "VERTEX-08-Pre-Publication",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "vertex-pre-publish",
        "method": "POST"
      }
    },
    {
      "name": "Check Reproducibility",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": {
        "command": "cd {{ $json.repo_path }} && python -m pytest tests/ && python scripts/validate_results.py"
      }
    },
    {
      "name": "Verify All Seeds",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const results = $json.experiment_results;\nconst seeds = [...new Set(results.map(r => r.seed))];\nconst required_seeds = 5;\nif (seeds.length < required_seeds) {\n  return { json: { valid: false, error: `Only ${seeds.length} seeds, need ${required_seeds}` } };\n}\nreturn { json: { valid: true, seeds } };"
      }
    },
    {
      "name": "Check Statistical Significance",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "{{ $json.stats_api_url }}",
        "body": {
          "baseline": "={{ $json.baseline_results }}",
          "treatment": "={{ $json.treatment_results }}",
          "alpha": 0.05
        }
      }
    },
    {
      "name": "Generate Checklist Report",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const checks = {\n  reproducibility: $('Check Reproducibility').item.json.exitCode === 0,\n  seeds: $('Verify All Seeds').item.json.valid,\n  significance: $('Check Statistical Significance').item.json.p_value < 0.05\n};\nconst allPassed = Object.values(checks).every(v => v);\nreturn { json: { checks, allPassed } };"
      }
    },
    {
      "name": "Send Final Report",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#research",
        "text": "📝 Pre-Publication Checklist\\n\\n{{ Object.entries($json.checks).map(([k,v]) => `${v ? '✅' : '❌'} ${k}`).join('\\n') }}\\n\\n{{ $json.allPassed ? '🎉 Ready to publish!' : '⚠️ Fix issues before publishing' }}"
      }
    }
  ]
}
```

---

## Python Integration Script

Add this to your experiment code to integrate with n8n:

```python
# src/utils/n8n_integration.py

import requests
import os
from datetime import datetime
from typing import Dict, Any, Optional

class VertexTracker:
    """Integration with n8n VERTEX-RESEARCH automation."""
    
    def __init__(self, webhook_base: str = None):
        self.webhook_base = webhook_base or os.getenv(
            'N8N_WEBHOOK_BASE', 
            'https://your-n8n-instance.com/webhook'
        )
        self.experiment_id = None
        self.notion_page_id = None
    
    def _send(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict]:
        """Send data to n8n webhook."""
        try:
            response = requests.post(
                f"{self.webhook_base}/{endpoint}",
                json=data,
                timeout=10
            )
            return response.json() if response.ok else None
        except Exception as e:
            print(f"n8n notification failed: {e}")
            return None
    
    def register_hypothesis(
        self,
        hypothesis: str,
        conditions: list,
        falsification_criteria: list,
        thresholds: Dict[str, float]
    ) -> Dict:
        """Register a hypothesis before experiments."""
        return self._send('vertex-hypothesis', {
            'hypothesis': hypothesis,
            'conditions': conditions,
            'falsification': falsification_criteria,
            'thresholds': thresholds,
            'timestamp': datetime.now().isoformat()
        })
    
    def start_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        git_commit: str
    ) -> None:
        """Log experiment start."""
        self.experiment_id = experiment_id
        response = self._send('vertex-experiment', {
            'event': 'start',
            'experiment_id': experiment_id,
            'config': config,
            'git_commit': git_commit,
            'timestamp': datetime.now().isoformat()
        })
        if response:
            self.notion_page_id = response.get('notion_page_id')
    
    def log_checkpoint(
        self,
        progress: float,
        current_seed: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log experiment checkpoint."""
        self._send('vertex-experiment', {
            'event': 'checkpoint',
            'experiment_id': self.experiment_id,
            'notion_page_id': self.notion_page_id,
            'progress': progress,
            'current_seed': current_seed,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def complete_experiment(self, results: Dict[str, Any]) -> None:
        """Log experiment completion."""
        self._send('vertex-experiment', {
            'event': 'complete',
            'experiment_id': self.experiment_id,
            'notion_page_id': self.notion_page_id,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_error(self, error_message: str) -> None:
        """Log experiment error."""
        self._send('vertex-experiment', {
            'event': 'error',
            'experiment_id': self.experiment_id,
            'notion_page_id': self.notion_page_id,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        })
    
    def request_gate_review(self, phase: int, artifacts: Dict[str, Any]) -> Dict:
        """Request automated gate review."""
        return self._send('vertex-gate-review', {
            'phase': phase,
            'artifacts': artifacts,
            'timestamp': datetime.now().isoformat()
        })


# Usage example
if __name__ == '__main__':
    tracker = VertexTracker()
    
    # Register hypothesis before experiments
    tracker.register_hypothesis(
        hypothesis="Entropy-based K selection reduces compute by >25%",
        conditions=["MoE architecture with learned router", "Calibrated thresholds"],
        falsification_criteria=["Compute savings < 15%", "Perplexity degradation > 2%"],
        thresholds={"min_savings": 0.25, "max_ppl_degradation": 0.02}
    )
    
    # Start experiment
    tracker.start_experiment(
        experiment_id="EXP-2026-01-19-001",
        config={"model": "mixtral", "k_values": [1, 2], "threshold": 1.275},
        git_commit="abc123"
    )
    
    # Log progress
    for seed in [42, 43, 44, 45, 46]:
        tracker.log_checkpoint(
            progress=(seed - 41) * 20,
            current_seed=seed,
            metrics={"avg_k": 1.38, "ppl": 3.87}
        )
    
    # Complete
    tracker.complete_experiment({
        "avg_k_mean": 1.38,
        "avg_k_std": 0.02,
        "compute_savings": 0.31,
        "ppl_change": 0.008
    })
```

---

## Environment Setup

### Required n8n Credentials

1. **Notion** - API key + Database IDs
2. **GitHub** - Personal Access Token
3. **Slack** - Webhook URL or OAuth
4. **OpenAI** - API key for GPT-4
5. **Email** - SMTP credentials

### Notion Database Structure

Create these databases:

| Database | Properties |
|----------|------------|
| **Projects** | Name, Phase, Status, Created, Last Gate |
| **Hypotheses** | ID, Statement, Status, Registration Date, Falsification Criteria |
| **Experiments** | ID, Project, Status, Start/End Time, Config, Results, Git Commit |
| **Papers** | Title, Authors, Year, Relevance, Status, Notes |
| **Gate Reviews** | Project, Phase, Date, Result, Missing Items |
| **Weekly Reports** | Date, Content, Key Metrics |

### n8n Docker Compose

```yaml
version: '3.8'
services:
  n8n:
    image: n8nio/n8n
    restart: always
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=n8n.vertexdata.it
      - N8N_PROTOCOL=https
      - WEBHOOK_URL=https://n8n.vertexdata.it/
    volumes:
      - n8n_data:/home/node/.n8n
      - /var/run/docker.sock:/var/run/docker.sock

volumes:
  n8n_data:
```

---

## Quick Start

1. **Deploy n8n** on your server
2. **Create Notion databases** with structure above
3. **Import workflows** from JSON files
4. **Set credentials** for all integrations
5. **Add Python tracker** to your experiment code
6. **Start new project** via webhook or n8n UI

---

## Workflow Summary

| # | Workflow | Trigger | Purpose |
|---|----------|---------|---------|
| 1 | Project Init | Manual/Webhook | Create project structure |
| 2 | Daily Standup | 9:00 daily | Morning priorities email |
| 3 | Literature Review | Phase 1 start | Automated paper search |
| 4 | Hypothesis Registration | Form submit | Validate & register hypotheses |
| 5 | Experiment Tracking | Python webhook | Log experiments in real-time |
| 6 | Gate Review | Phase complete | Automated quality check |
| 7 | Weekly Report | Friday 17:00 | Weekly summary email |
| 8 | Pre-Publication | Paper ready | Final checklist before publish |

---

*VERTEX-RESEARCH Protocol Automation v1.0*
