from app.advanced import GraphRequest, execute_graph


def test_real_graph_executes_and_checks_handoffs():
    result = execute_graph(GraphRequest(question='retrieval', corpus=[
        {'id': 'doc1', 'text': 'retrieval finds evidence'}]))
    assert result['passed'] is True
    assert [event['agent'] for event in result['trace']] == [
        'planner', 'retriever', 'analyst', 'verifier']
    assert '[doc1]' in result['answer']
    assert all(result['handoff_checks'].values())


def test_missing_required_handoff_fails_gate():
    result = execute_graph(GraphRequest(question='retrieval', corpus=[
        {'id': 'doc1', 'text': 'retrieval finds evidence'}],
        required_handoffs=['planner->verifier']))
    assert result['verified'] is True
    assert result['passed'] is False


def test_celery_task_executes_locally_without_broker():
    from app.jobs import graph_task
    result = graph_task.apply(args=[{'question': 'retrieval', 'corpus': [
        {'id': 'doc1', 'text': 'retrieval finds evidence'}]}], throw=True).get()
    assert result['passed'] is True
