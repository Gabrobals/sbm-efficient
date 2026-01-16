1. Sistema di Verifica degli OutputArchitettura di Verifica Multi-Livellopythonclass OutputVerificationSystem:
    def __init__(self):
        self.verification_layers = [
            SchemaValidator(),
            FactualityChecker(),
            ConsistencyVerifier(),
            ConfidenceScorer(),
            HumanReviewRouter()
        ]
    
    async def verify_output(self, output, context):
        verification_results = {
            'passed': True,
            'confidence': 1.0,
            'issues': [],
            'requires_human_review': False
        }
        
        # Layer 1: Schema Validation
        schema_result = await self.validate_schema(output)
        if not schema_result.valid:
            verification_results['passed'] = False
            verification_results['issues'].append({
                'layer': 'schema',
                'severity': 'critical',
                'details': schema_result.errors
            })
            return verification_results
        
        # Layer 2: Factuality Check (cross-reference with sources)
        factuality_result = await self.check_factuality(output, context.sources)
        verification_results['confidence'] *= factuality_result.confidence
        
        # Layer 3: Self-Consistency (multiple sampling)
        consistency_result = await self.verify_consistency(output, context)
        verification_results['confidence'] *= consistency_result.score
        
        # Layer 4: Critic Model
        critic_result = await self.critic_model.evaluate(output, context)
        
        # Routing logic
        if verification_results['confidence'] < 0.85:
            verification_results['requires_human_review'] = True
        
        return verification_resultsModalità di Errore Principali1. Allucinazioni Fattuali

Dati numerici inventati in contesti finanziari
Citazioni inesistenti da documenti fonte
Mitigazione: Cross-reference con grafi di conoscenza, entity linking, citation verification
2. Inconsistenze Logiche

Contraddizioni tra sezioni diverse del report
Calcoli che non tornano con i dati presentati
Mitigazione: Constraint satisfaction checking, symbolic reasoning layer
3. Degrado dello Schema

Output che violano il formato JSON richiesto
Campi mancanti in strutture dati complesse
Mitigazione: Constrained decoding (grammar-based), Pydantic validation, retry con error feedback
4. Deriva Contestuale

Perdita di contesto in reasoning chain lunghi
Riferimenti ambigui in analisi multi-documento
Mitigazione: Context compression, sliding window con overlap, entity tracking
5. Confidenza Calibrazione

Overconfidence su output errati
Mitigazione: Temperature scaling, ensembling, calibration datasets
2. Modelli di Autocoerenza vs Ricompensa di Processo vs CriticaSelf-Consistency ModelsMeccanismo: Sample multipli con temperature>0, majority voting o clustering semanticopythonasync def self_consistency_verification(query, n_samples=5):
    responses = await asyncio.gather(*[
        model.generate(query, temperature=0.7) 
        for _ in range(n_samples)
    ])
    
    # Semantic clustering
    embeddings = [embed(r) for r in responses]
    clusters = cluster_responses(embeddings, threshold=0.85)
    
    # Majority cluster = consensus answer
    consensus = max(clusters, key=lambda c: len(c.members))
    
    return {
        'answer': consensus.representative,
        'confidence': len(consensus.members) / n_samples,
        'agreement_score': consensus.cohesion
    }Quando usarlo:

Reasoning matematico/logico dove esiste una risposta corretta
Query con vincoli verificabili (es. SQL generation)
Non adatto: Compiti creativi, analisi soggettive
Process Reward Models (PRM)Meccanismo: Modello addestrato a valutare la correttezza di ogni step in una chain-of-thoughtpythonclass ProcessRewardModel:
    def score_reasoning_chain(self, steps, ground_truth=None):
        step_scores = []
        cumulative_validity = 1.0
        
        for i, step in enumerate(steps):
            context = steps[:i]  # Previous steps
            score = self.prm_model.evaluate_step(
                step=step,
                context=context,
                expected_outcome=ground_truth
            )
            step_scores.append(score)
            cumulative_validity *= score
            
            # Early stopping se lo step è chiaramente errato
            if score < 0.3:
                return {
                    'valid': False,
                    'failed_at_step': i,
                    'step_scores': step_scores
                }
        
        return {
            'valid': cumulative_validity > 0.7,
            'step_scores': step_scores,
            'overall_score': cumulative_validity
        }Quando usarlo:

Reasoning multi-step verificabile (matematica, codice)
Training con verifier models (es. PRM800K dataset)
Vantaggio: Identifica esattamente dove il reasoning fallisce
Critic ModelsMeccanismo: LLM dedicato che valuta completezza, accuratezza, stile dell'output principalepythonclass CriticModel:
    async def critique(self, output, criteria):
        critique_prompt = f"""
        Valuta questo output finanziario secondo questi criteri:
        1. Accuratezza fattuale (cita le fonti)
        2. Completezza dell'analisi
        3. Conformità agli standard di reporting
        4. Logica e coerenza
        
        Output: {output}
        
        Fornisci punteggi (0-10) e spiegazioni dettagliate.
        """
        
        critique = await self.critic_llm.generate(
            critique_prompt,
            response_format=CritiqueSchema
        )
        
        # Iterative refinement
        if critique.overall_score < 8:
            refined_output = await self.refine_with_feedback(
                original=output,
                feedback=critique.detailed_feedback
            )
            return refined_output
        
        return outputQuando usarlo:

Output complessi e soggettivi (reports, analisi narrative)
Quando servono feedback qualitativi dettagliati
Iterative refinement loops (critica → revisione → critica)
Confronto e SceltaCriterioSelf-ConsistencyPRMCriticTipo di taskReasoning oggettivoMulti-step chainOutput complessiCosto computazionaleAlto (N samples)MedioMedio-AltoGranularità feedbackBassaAlta (per-step)Molto altaTraining richiestoNoSì (dataset labeled)No (zero-shot)Uso idealeMath, QACode, proofsReports, analisiNel contesto finanziario:

PRM: Per pipeline di calcolo multi-step (DCF models, ratio analysis)
Self-Consistency: Per estrazione dati da documenti (verificare che 5 sampling estraggono lo stesso numero)
Critic: Per review di report finali prima della pubblicazione
3. Sistema con Accuratezza 95% ed Errori CatastroficiArchitettura Defense-in-Depthpythonclass HighStakesPipeline:
    """
    Design principle: FAIL SAFE, not fail silent
    """
    
    def __init__(self):
        # Multiple independent verification systems
        self.primary_model = GPT4()
        self.shadow_model = Claude()  # Independent verification
        self.rule_engine = SymbolicValidator()
        self.human_review_queue = ReviewQueue()
        
        # Cascading confidence thresholds
        self.thresholds = {
            'auto_approve': 0.98,
            'secondary_check': 0.95,
            'human_review': 0.90,
            'reject': 0.0
        }
    
    async def process_high_stakes_request(self, request):
        # Stage 1: Dual model generation
        primary_output, shadow_output = await asyncio.gather(
            self.primary_model.generate(request),
            self.shadow_model.generate(request)
        )
        
        # Stage 2: Agreement check
        agreement_score = self.semantic_similarity(
            primary_output, shadow_output
        )
        
        if agreement_score < 0.9:
            # Models disagree - human review mandatory
            return self.route_to_human(
                reason="Model disagreement",
                outputs=[primary_output, shadow_output]
            )
        
        # Stage 3: Symbolic validation (hard constraints)
        rule_check = self.rule_engine.validate(primary_output)
        if not rule_check.passed:
            # Hard constraint violation - automatic rejection
            return self.reject_with_explanation(rule_check.violations)
        
        # Stage 4: Confidence-based routing
        confidence = primary_output.confidence_score
        
        if confidence >= self.thresholds['auto_approve']:
            # Still log for audit trail
            self.audit_log.record(primary_output, auto_approved=True)
            return primary_output
        
        elif confidence >= self.thresholds['secondary_check']:
            # Additional verification layer
            verified = await self.deep_verification(primary_output)
            if verified:
                return primary_output
            else:
                return self.route_to_human(reason="Failed deep verification")
        
        else:
            # Below threshold - mandatory human review
            return self.route_to_human(
                reason=f"Low confidence: {confidence:.2f}"
            )Strategie Anti-Catastrofiche1. Constrained Decoding
python# Per output finanziari: vincoli grammaticali hard-coded
from outlines import models, generate

model = models.transformers("gpt-4")
schema = {
    "type": "object",
    "properties": {
        "revenue": {"type": "number", "minimum": 0},
        "growth_rate": {"type": "number", "minimum": -100, "maximum": 1000}
    },
    "required": ["revenue", "growth_rate"]
}

# Il modello NON PUÒ generare output che violano lo schema
generator = generate.json(model, schema)
result = generator("Analizza i risultati trimestrali...")2. Canary Queries
python# Inserire test questions con risposta nota nel flusso
class CanarySystem:
    def inject_canaries(self, user_query_batch):
        canaries = [
            {"query": "What is 2+2?", "expected": "4"},
            {"query": "Capital of France?", "expected": "Paris"}
        ]
        # Mix canaries con query reali
        mixed_batch = user_query_batch + canaries
        random.shuffle(mixed_batch)
        return mixed_batch
    
    def validate_system_health(self, results):
        canary_results = [r for r in results if r.is_canary]
        accuracy = sum(r.correct for r in canary_results) / len(canary_results)
        
        if accuracy < 0.95:
            # Sistema degraded - stop processing
            self.emergency_shutdown()3. Ensemble con Veto Power
pythonclass VetoEnsemble:
    def __init__(self):
        # Modelli specializzati con capacità di veto
        self.specialist_models = {
            'numerical': FinancialCalculatorModel(),
            'factual': KnowledgeGraphVerifier(),
            'regulatory': ComplianceChecker()
        }
    
    async def generate_with_veto(self, query):
        primary_output = await self.primary_model.generate(query)
        
        # Ogni specialist può bloccare l'output
        for name, specialist in self.specialist_models.items():
            veto_check = await specialist.verify(primary_output, query)
            
            if veto_check.veto:
                # VETO esercitato - output bloccato
                return {
                    'approved': False,
                    'vetoed_by': name,
                    'reason': veto_check.reason,
                    'requires_human': True
                }
        
        return {'approved': True, 'output': primary_output}4. Graceful Degradation
python# Invece di fallire, restringi lo scope
class GracefulDegrader:
    async def handle_uncertain_output(self, output, confidence):
        if confidence < 0.85:
            # Invece di dare risposta completa, fornisci partial answer
            return {
                'type': 'partial_answer',
                'confident_facts': self.extract_high_confidence_claims(output),
                'uncertain_areas': self.extract_low_confidence_claims(output),
                'recommendation': 'human_review_recommended',
                'explanation': 'Alcune analisi richiedono verifica umana'
            }5. Monitoring Real-Time
python# Anomaly detection su distribuzione degli output
class OutputMonitor:
    def __init__(self):
        self.baseline_stats = self.compute_baseline()
    
    def check_distribution_shift(self, recent_outputs):
        current_stats = self.compute_stats(recent_outputs)
        
        # KL divergence tra baseline e current
        divergence = kl_divergence(self.baseline_stats, current_stats)
        
        if divergence > threshold:
            # Distribution shift detected - possibile model degradation
            self.alert_ops_team()
            self.enable_enhanced_verification()4. Sistema Multi-Agente: StockLoop Analytical EngineCaso Reale: Sistema di Analisi per Retail OtticoArchitettura: Orchestrazione di 4 agenti specializzati per produrre raccomandazioni di stock managementpythonclass StockLoopMultiAgent:
    def __init__(self):
        self.agents = {
            'data_collector': DataCollectorAgent(),
            'analyzer': AnalyticalAgent(),
            'forecaster': ForecastingAgent(),
            'recommender': RecommendationAgent()
        }
        
        self.coordinator = CoordinatorAgent(
            workflow_graph=self.build_workflow_graph()
        )
        
        self.state_manager = SharedStateManager()
        self.failure_handler = FailureRecoverySystem()
    
    def build_workflow_graph(self):
        """
        DAG del workflow:
        DataCollector → Analyzer → Forecaster → Recommender
                     ↓
                  Validator (parallel)
        """
        graph = WorkflowGraph()
        
        graph.add_edge('data_collector', 'analyzer')
        graph.add_edge('data_collector', 'validator')
        graph.add_edge('analyzer', 'forecaster')
        graph.add_edge('forecaster', 'recommender')
        graph.add_edge('validator', 'recommender')
        
        return graphCoordinamento: State Machine Patternpythonclass CoordinatorAgent:
    async def orchestrate_analysis(self, shop_id, time_period):
        # Shared state accessibile a tutti gli agenti
        shared_state = {
            'shop_id': shop_id,
            'time_period': time_period,
            'status': 'initializing',
            'results': {},
            'errors': []
        }
        
        try:
            # Phase 1: Data Collection (con timeout)
            async with timeout(30):
                sales_data = await self.agents['data_collector'].collect(
                    shared_state
                )
                shared_state['results']['sales_data'] = sales_data
                shared_state['status'] = 'data_collected'
            
            # Phase 2: Parallel execution (Analyzer + Validator)
            analyzer_task = self.agents['analyzer'].analyze(shared_state)
            validator_task = self.agents['validator'].validate(shared_state)
            
            analysis, validation = await asyncio.gather(
                analyzer_task, 
                validator_task,
                return_exceptions=True  # Non bloccare se uno fallisce
            )
            
            # Gestione fallimenti parziali
            if isinstance(analysis, Exception):
                await self.failure_handler.handle_agent_failure(
                    agent='analyzer',
                    error=analysis,
                    shared_state=shared_state
                )
                # Fallback: usa analisi semplificata
                analysis = await self.fallback_analysis(shared_state)
            
            shared_state['results']['analysis'] = analysis
            shared_state['results']['validation'] = validation
            shared_state['status'] = 'analyzed'
            
            # Phase 3: Forecasting (dipende da Phase 2)
            forecast = await self.agents['forecaster'].forecast(shared_state)
            shared_state['results']['forecast'] = forecast
            
            # Phase 4: Final Recommendations
            recommendations = await self.agents['recommender'].recommend(
                shared_state
            )
            
            return {
                'success': True,
                'recommendations': recommendations,
                'metadata': self.extract_metadata(shared_state)
            }
            
        except Exception as e:
            # Top-level failure handling
            return await self.failure_handler.handle_orchestration_failure(
                error=e,
                shared_state=shared_state
            )Gestione Guasti: Circuit Breaker + Retry con Backoffpythonclass FailureRecoverySystem:
    def __init__(self):
        self.circuit_breakers = {
            agent: CircuitBreaker(
                failure_threshold=3,
                timeout=60,
                recovery_timeout=300
            )
            for agent in ['data_collector', 'analyzer', 'forecaster', 'recommender']
        }
        
        self.retry_policies = {
            'data_collector': RetryPolicy(max_attempts=3, backoff='exponential'),
            'analyzer': RetryPolicy(max_attempts=2, backoff='linear'),
            'forecaster': RetryPolicy(max_attempts=2, backoff='exponential'),
            'recommender': RetryPolicy(max_attempts=1, backoff=None)
        }
    
    async def handle_agent_failure(self, agent, error, shared_state):
        logger.error(f"Agent {agent} failed: {error}")
        
        # Check circuit breaker
        if self.circuit_breakers[agent].is_open():
            # Circuit open: non tentare retry, usa fallback immediato
            return await self.execute_fallback(agent, shared_state)
        
        # Retry con backoff
        retry_policy = self.retry_policies[agent]
        
        for attempt in range(retry_policy.max_attempts):
            try:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                
                # Retry con fresh context
                result = await self.retry_agent_execution(
                    agent, shared_state, attempt
                )
                
                # Success: reset circuit breaker
                self.circuit_breakers[agent].record_success()
                return result
                
            except Exception as retry_error:
                self.circuit_breakers[agent].record_failure()
                
                if attempt == retry_policy.max_attempts - 1:
                    # Max retries reached: fallback
                    return await self.execute_fallback(agent, shared_state)
    
    async def execute_fallback(self, agent, shared_state):
        """
        Fallback strategies per ogni agente
        """
        fallbacks = {
            'data_collector': self.use_cached_data,
            'analyzer': self.use_simplified_analysis,
            'forecaster': self.use_historical_average,
            'recommender': self.use_conservative_recommendations
        }
        
        return await fallbacks[agent](shared_state)Debug: Observability Stackpythonclass MultiAgentDebugger:
    def __init__(self):
        self.tracer = DistributedTracer()  # OpenTelemetry
        self.event_store = EventStore()
        self.visualizer = WorkflowVisualizer()
    
    def instrument_agent(self, agent_name, agent_func):
        """
        Decorator per tracciare esecuzione agenti
        """
        @wraps(agent_func)
        async def wrapper(*args, **kwargs):
            span = self.tracer.start_span(f"agent.{agent_name}")
            
            # Log input
            self.event_store.record({
                'timestamp': time.time(),
                'agent': agent_name,
                'type': 'input',
                'data': self.serialize_inputs(args, kwargs)
            })
            
            try:
                result = await agent_func(*args, **kwargs)
                
                # Log output
                self.event_store.record({
                    'timestamp': time.time(),
                    'agent': agent_name,
                    'type': 'output',
                    'data': self.serialize_output(result),
                    'latency': span.duration
                })
                
                span.set_status("success")
                return result
                
            except Exception as e:
                # Log error con stack trace completo
                self.event_store.record({
                    'timestamp': time.time(),
                    'agent': agent_name,
                    'type': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'shared_state_snapshot': self.snapshot_state(kwargs.get('shared_state'))
                })
                
                span.set_status("error")
                span.record_exception(e)
                raise
            
            finally:
                span.end()
        
        return wrapper
    
    def replay_failed_workflow(self, workflow_id):
        """
        Time-travel debugging: replay workflow da eventi registrati
        """
        events = self.event_store.get_events(workflow_id)
        
        # Ricostruisci stato a ogni step
        for i, event in enumerate(events):
            print(f"\n=== Step {i}: {event['agent']} ===")
            print(f"Input: {event.get('data')}")
            
            if event['type'] == 'error':
                print(f"ERROR: {event['error']}")
                print(f"State snapshot: {event['shared_state_snapshot']}")
                
                # Interactive debugging
                self.visualizer.show_workflow_state(event['shared_state_snapshot'])
                breakpoint()  # Permette ispezione interattivaPattern di Comunicazione: Event Buspythonclass AgentEventBus:
    """
    Pub/Sub pattern per comunicazione asincrona tra agenti
    """
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_log = []
    
    def subscribe(self, event_type, agent_callback):
        self.subscribers[event_type].append(agent_callback)
    
    async def publish(self, event_type, payload):
        self.event_log.append({
            'timestamp': time.time(),
            'type': event_type,
            'payload': payload
        })
        
        # Notify subscribers in parallel
        callbacks = self.subscribers[event_type]
        await asyncio.gather(*[
            callback(payload) for callback in callbacks
        ])

# Uso:
event_bus = AgentEventBus()

# Forecaster si subscribe a nuove analisi
event_bus.subscribe(
    'analysis_complete',
    lambda payload: forecaster.update_forecast(payload)
)

# Analyzer pubblica risultato
await event_bus.publish('analysis_complete', analysis_result)5. Osservabilità e Debug in Sistemi Multi-StepStack di Osservabilità Completopythonclass LLMObservabilityPlatform:
    """
    Ispirato a LangSmith, Arize, HoneyHive
    """
    def __init__(self):
        # Distributed tracing
        self.tracer = OpenTelemetryTracer()
        
        # Structured logging
        self.logger = StructuredLogger(
            outputs=['console', 'file', 'elasticsearch']
        )
        
        # Metrics collection
        self.metrics = PrometheusMetrics([
            'llm_latency',
            'token_usage',
            'error_rate',
            'confidence_score'
        ])
        
        # Trace storage
        self.trace_db = TraceDatabase()
        
        # Visualization
        self.dashboard = GrafanaDashboard()
    
    def trace_workflow(self, workflow_name):
        """
        Decorator per tracciare interi workflow
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Root span per l'intero workflow
                with self.tracer.start_as_current_span(
                    f"workflow.{workflow_name}"
                ) as span:
                    span.set_attribute("workflow.name", workflow_name)
                    span.set_attribute("workflow.input", str(args))
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("workflow.status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("workflow.status", "failed")
                        span.record_exception(e)
                        raise
            
            return wrapper
        return decoratorCaso d'Uso: Debug di Workflow a 15 Step Fallito all'11°pythonclass WorkflowDebugger:
    """
    Sistema per debug granulare di pipeline LLM complesse
    """
    
    async def debug_failed_workflow(self, trace_id):
        """
        Analizza un workflow fallito usando il trace_id
        """
        # 1. Recupera il trace completo
        trace = await self.trace_db.get_trace(trace_id)
        
        print(f"=== Workflow Trace: {trace_id} ===")
        print(f"Total steps: {len(trace.spans)}")
        print(f"Failed at step: {trace.failed_step_index}")
        print(f"Error: {trace.error_message}\n")
        
        # 2. Visualizza la cascade di chiamate
        self.visualize_trace_waterfall(trace)
        
        # 3. Analisi del passo fallito (step 11)
        failed_span = trace.spans[trace.failed_step_index]
        
        print(f"\n=== Failed Step Analysis (Step 11) ===")
        print(f"Step name: {failed_span.name}")
        print(f"Duration: {failed_span.duration}ms")
        print(f"Input tokens: {failed_span.attributes['input_tokens']}")
        print(f"Output tokens: {failed_span.attributes['output_tokens']}")
        
        # 4. Ispeziona input/output del passo fallito
        print(f"\nInput prompt:")
        print(failed_span.attributes['prompt'][:500] + "...")
        
        print(f"\nPartial output before failure:")
        print(failed_span.attributes['partial_output'])
        
        # 5. Controlla il contesto: cosa è successo nei passi precedenti?
        print(f"\n=== Context from Previous Steps ===")
        for i in range(max(0, trace.failed_step_index - 3), trace.failed_step_index):
            prev_span = trace.spans[i]
            print(f"Step {i}: {prev_span.name}")
            print(f"  - Status: {prev_span.status}")
            print(f"  - Output summary: {prev_span.attributes['output_summary']}")
        
        # 6. Root cause analysis
        root_cause = await self.analyze_root_cause(trace, failed_span)
        print(f"\n=== Root Cause Analysis ===")
        print(root_cause.explanation)
        print(f"Likely causes: {root_cause.likely_causes}")
        print(f"Suggested fixes: {root_cause.suggested_fixes}")
        
        # 7. Replay con modifiche
        print(f"\n=== Attempting Replay with Fixes ===")
        replay_result = await self.replay_with_fixes(trace, root_cause.suggested_fixes[0])
        
        return {
            'trace': trace,
            'root_cause': root_cause,
            'replay_result': replay_result
        }
    
    def visualize_trace_waterfall(self, trace):
        """
        Visualizzazione waterfall come Chrome DevTools
        """
        print("\n=== Execution Timeline ===")
        max_duration = max(span.duration for span in trace.spans)
        
        for i, span in enumerate(trace.spans):
            # Scala visuale
            bar_length = int((span.duration / max_duration) * 50)
            bar = "█" * bar_length
            
            status_icon = "✓" if span.status == "success" else "✗"
            
            print(f"{i:2d}. {status_icon} {span.name:30s} {bar} {span.duration:6.0f}ms")
            
            if span.status == "error":
                print(f"     ERROR: {span.error_message}")
    
    async def analyze_root_cause(self, trace, failed_span):
        """
        Usa un LLM per analizzare il fallimento
        """
        analysis_prompt = f"""
        Analizza questo fallimento in un workflow LLM:
        
        Step fallito: {failed_span.name} (step {trace.failed_step_index} di {len(trace.spans)})
        Errore: {failed_span.error_message}
        
        Input del passo: {failed_span.attributes['prompt']}
        Output parziale: {failed_span.attributes.get('partial_output', 'N/A')}
        
        Contesto dai passi precedenti:
        {self.format_previous_steps_context(trace, trace.failed_step_index)}
        
        Identifica:
        1. La causa radice del fallimento
        2. Se è un problema di prompt, dati, o modello
        3. Suggerimenti concreti per fixare il problema
        """
        
        analysis = await self.diagnostic_llm.generate(
            analysis_prompt,
            response_format=RootCauseAnalysis
        )
        
        return analysis
    
    async def replay_with_fixes(self, trace, fix):
        """
        Replay del workflow applicando una fix suggerita
        """
        # Ricostruisci il workflow fino al passo fallito
        workflow_state = self.reconstruct_state_at_step(
            trace, 
            trace.failed_step_index - 1
        )
        
        # Applica la fix
        if fix.type == 'prompt_modification':
            modified_prompt = fix.apply_to_prompt(
                trace.spans[trace.failed_step_index].attributes['prompt']
            )
            
            # Re-esegui dal passo modificato
            result = await self.rerun_from_step(
                workflow_state,
                trace.failed_step_index,
                modified_prompt=modified_prompt
            )
            
            return resultStrumenti di Debug Specifici1. Diff Viewer per Prompts
pythonclass PromptDiffAnalyzer:
    def compare_prompt_versions(self, trace_id_1, trace_id_2, step_index):
        """
        Confronta lo stesso passo in due esecuzioni diverse
        """
        trace1 = self.trace_db.get_trace(trace_id_1)
        trace2 = self.trace_db.get_trace(trace_id_2)
        
        prompt1 = trace1.spans[step_index].attributes['prompt']
        prompt2 = trace2.spans[step_index].attributes['prompt']
        
        # Unified diff
        diff = difflib.unified_diff(
            prompt1.splitlines(),
            prompt2.splitlines(),
            lineterm=''
        )
        
        print('\n'.join(diff))
        
        # Semantic diff (usando embeddings)
        semantic_similarity = self.compute_semantic_similarity(prompt1, prompt2)
        print(f"\nSemantic similarity: {semantic_similarity:.2%}")2. Token Usage Analyzer
pythonclass TokenUsageAnalyzer:
    def analyze_trace_efficiency(self, trace):
        """
        Identifica sprechi di token nel workflow
        """
        total_tokens = 0
        token_breakdown = []
        
        for span in trace.spans:
            step_tokens = (
                span.attributes['input_tokens'] +
                span.attributes['output_tokens']
            )
            total_tokens += step_tokens
            
            token_breakdown.append({
                'step': span.name,
                'tokens': step_tokens,
                'percentage': 0  # Calcolato dopo
            })
        
        # Calcola percentuali
        for item in token_breakdown:
            item['percentage'] = (item['tokens'] / total_tokens) * 100
        
        # Identifica outliers
        avg_tokens = total_tokens / len(trace.spans)
        outliers = [
            item for item in token_breakdown
            if item['tokens'] > avg_tokens * 2
        ]
        
        return {
            'total_tokens': total_tokens,
            'breakdown': token_breakdown,
            'outliers': outliers,
            'suggestions': self.suggest_optimizations(outliers)
        }
    
    def suggest_optimizations(self, outliers):
        suggestions = []
        for outlier in outliers:
            suggestions.append(f"""
            Step '{outlier['step']}' usa {outlier['tokens']} tokens ({outlier['percentage']:.1f}%).
            Considera:
            - Prompt compression
            - Rimuovere esempi ridondanti
            - Splittare in sub-tasks più piccoli
            """)
        return suggestions3. Confidence Calibration Monitor
pythonclass ConfidenceMonitor:
    def analyze_confidence_accuracy(self, trace):
        """
        Verifica se i confidence score sono calibrati
        """
        calibration_data = []
        
        for span in trace.spans:
            if 'confidence_score' in span.attributes:
                calibration_data.append({
                    'step': span.name,
                    'confidence': span.attributes['confidence_score'],
                    'actual_correct': span.attributes.get('verified_correct', None)
                })
        
        # Plotta calibration curve
        self.plot_calibration_curve(calibration_data)
        
        # Calcola Expected Calibration Error
        ece = self.compute_ece(calibration_data)
        print(f"Expected Calibration Error: {ece:.3f}")
        
        if ece > 0.1:
            print("⚠️  Warning: Model confidence non calibrato!")
            print("Considera: temperature scaling, Platt scaling")