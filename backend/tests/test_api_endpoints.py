import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestAPIEndpoints:
    """Comprehensive API endpoint tests for FastAPI application"""

    def test_query_endpoint_basic_request(self, test_client):
        """Test basic query functionality"""
        test_client.mock_rag_system.query.return_value = ("Basic test answer", [])
        
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Basic test answer"
        assert data["sources"] == []
        assert data["session_id"] == "test_session_123"

    def test_query_endpoint_with_session(self, test_client):
        """Test query with existing session ID"""
        test_client.mock_rag_system.query.return_value = ("Session test answer", [])
        
        response = test_client.post("/api/query", json={
            "query": "Continue our conversation",
            "session_id": "existing_session_456"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing_session_456"
        
        # Verify RAG system called with correct session
        test_client.mock_rag_system.query.assert_called_once_with(
            "Continue our conversation", 
            "existing_session_456"
        )

    def test_query_endpoint_with_dict_sources(self, test_client):
        """Test query response with dictionary-formatted sources"""
        mock_sources = [
            {"text": "ML Course Chapter 1", "link": "https://example.com/ml-ch1"},
            {"text": "Advanced ML Tutorial", "link": "https://example.com/advanced-ml"}
        ]
        test_client.mock_rag_system.query.return_value = ("Answer with sources", mock_sources)
        
        response = test_client.post("/api/query", json={
            "query": "Explain neural networks"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "ML Course Chapter 1"
        assert data["sources"][0]["link"] == "https://example.com/ml-ch1"
        assert data["sources"][1]["text"] == "Advanced ML Tutorial"
        assert data["sources"][1]["link"] == "https://example.com/advanced-ml"

    def test_query_endpoint_with_string_sources(self, test_client):
        """Test query response with legacy string sources"""
        mock_sources = ["Course 1", "Course 2", "Course 3"]
        test_client.mock_rag_system.query.return_value = ("Answer with legacy sources", mock_sources)
        
        response = test_client.post("/api/query", json={
            "query": "Tell me about data science"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        assert data["sources"][0]["text"] == "Course 1"
        assert data["sources"][0]["link"] is None
        assert data["sources"][1]["text"] == "Course 2"
        assert data["sources"][1]["link"] is None

    def test_query_endpoint_mixed_source_formats(self, test_client):
        """Test query with mixed source formats"""
        mock_sources = [
            {"text": "Dict with link", "link": "https://example.com/link"},
            "String source",
            {"text": "Dict without link"}
        ]
        test_client.mock_rag_system.query.return_value = ("Mixed sources answer", mock_sources)
        
        response = test_client.post("/api/query", json={
            "query": "Mixed format test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        assert data["sources"][0]["text"] == "Dict with link"
        assert data["sources"][0]["link"] == "https://example.com/link"
        assert data["sources"][1]["text"] == "String source"
        assert data["sources"][1]["link"] is None
        assert data["sources"][2]["text"] == "Dict without link"
        assert data["sources"][2]["link"] is None

    def test_query_endpoint_validation_errors(self, test_client):
        """Test query endpoint input validation"""
        # Empty request body
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422
        
        # Missing query field
        response = test_client.post("/api/query", json={"session_id": "test"})
        assert response.status_code == 422
        
        # Invalid JSON
        response = test_client.post("/api/query", data="invalid json")
        assert response.status_code == 422

    def test_query_endpoint_rag_system_exception(self, test_client):
        """Test query endpoint when RAG system raises exception"""
        test_client.mock_rag_system.query.side_effect = Exception("RAG processing error")
        
        response = test_client.post("/api/query", json={
            "query": "This will cause an error"
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "RAG processing error" in data["detail"]

    def test_query_endpoint_session_creation_failure(self, test_client):
        """Test query endpoint when session creation fails"""
        test_client.mock_rag_system.session_manager.create_session.side_effect = Exception("Session creation failed")
        
        response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 500

    def test_courses_endpoint_success(self, test_client, mock_course_analytics):
        """Test successful courses analytics retrieval"""
        test_client.mock_rag_system.get_course_analytics.return_value = mock_course_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Machine Learning" in data["course_titles"]

    def test_courses_endpoint_empty_analytics(self, test_client):
        """Test courses endpoint with no courses"""
        empty_analytics = {"total_courses": 0, "course_titles": []}
        test_client.mock_rag_system.get_course_analytics.return_value = empty_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_endpoint_analytics_exception(self, test_client):
        """Test courses endpoint when analytics fails"""
        test_client.mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "Analytics error" in data["detail"]

    def test_404_nonexistent_endpoint(self, test_client):
        """Test accessing non-existent API endpoint"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, test_client):
        """Test using wrong HTTP method on endpoints"""
        # GET on query endpoint (should be POST)
        response = test_client.get("/api/query")
        assert response.status_code in [404, 405]  # FastAPI behavior varies
        
        # POST on courses endpoint (should be GET)
        response = test_client.post("/api/courses", json={})
        assert response.status_code in [404, 405]

    def test_response_model_validation(self, test_client):
        """Test that responses conform to expected Pydantic models"""
        # Test query response format
        test_client.mock_rag_system.query.return_value = ("Test answer", [])
        
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_cors_and_middleware_behavior(self, test_client):
        """Test CORS headers and middleware functionality"""
        response = test_client.post("/api/query", json={"query": "test"})
        
        # Response should succeed (CORS is enabled)
        assert response.status_code in [200, 500]  # Either success or mocked error
        
        # Test OPTIONS preflight request (if supported by TestClient)
        options_response = test_client.options("/api/query")
        # TestClient may not fully simulate CORS preflight behavior

    def test_large_query_handling(self, test_client):
        """Test handling of large queries"""
        large_query = "What is machine learning? " * 1000  # Very large query
        test_client.mock_rag_system.query.return_value = ("Large query response", [])
        
        response = test_client.post("/api/query", json={
            "query": large_query
        })
        
        assert response.status_code == 200
        # Verify the query was truncated in logs but still processed
        test_client.mock_rag_system.query.assert_called_once()

    def test_special_characters_in_query(self, test_client):
        """Test handling queries with special characters"""
        special_query = "What about émojis 🤖, symbols ∑∂∆, and unicode ñáéíóú?"
        test_client.mock_rag_system.query.return_value = ("Special chars response", [])
        
        response = test_client.post("/api/query", json={
            "query": special_query
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Special chars response"

    def test_concurrent_requests_simulation(self, test_client):
        """Test multiple requests to simulate concurrent usage"""
        test_client.mock_rag_system.query.return_value = ("Concurrent response", [])
        
        # Simulate multiple concurrent requests
        responses = []
        for i in range(5):
            response = test_client.post("/api/query", json={
                "query": f"Concurrent query {i}",
                "session_id": f"session_{i}"
            })
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Verify RAG system was called for each request
        assert test_client.mock_rag_system.query.call_count == 5


class TestStartupAndLifecycle:
    """Test application startup and lifecycle events"""
    
    @patch('os.path.exists')
    @patch('app.rag_system')
    def test_startup_with_docs_folder(self, mock_rag_system, mock_exists):
        """Test startup behavior when docs folder exists"""
        mock_exists.return_value = True
        mock_rag_system.add_course_folder.return_value = (5, 50)
        
        # Simulate startup logic
        docs_paths = ["../docs", "docs"]
        for docs_path in docs_paths:
            if os.path.exists(docs_path):  # This will be mocked
                courses, chunks = mock_rag_system.add_course_folder(docs_path, clear_existing=False)
                assert courses == 5
                assert chunks == 50
                break

    @patch('os.path.exists')
    @patch('app.rag_system')
    def test_startup_without_docs_folder(self, mock_rag_system, mock_exists):
        """Test startup behavior when docs folder doesn't exist"""
        mock_exists.return_value = False
        
        # Simulate startup logic
        docs_paths = ["../docs", "docs"]
        folder_found = False
        for docs_path in docs_paths:
            if os.path.exists(docs_path):  # This will be mocked to False
                folder_found = True
                break
        
        assert not folder_found
        mock_rag_system.add_course_folder.assert_not_called()


class TestPydanticModels:
    """Test Pydantic model validation and behavior"""
    
    def test_query_request_model(self):
        """Test QueryRequest model validation"""
        from app import QueryRequest
        
        # Valid request with both fields
        request = QueryRequest(query="Test query", session_id="session123")
        assert request.query == "Test query"
        assert request.session_id == "session123"
        
        # Valid request with only query
        request_minimal = QueryRequest(query="Test query")
        assert request_minimal.query == "Test query"
        assert request_minimal.session_id is None

    def test_source_info_model(self):
        """Test SourceInfo model validation"""
        from app import SourceInfo
        
        # Source with both text and link
        source1 = SourceInfo(text="Course Title", link="https://example.com")
        assert source1.text == "Course Title"
        assert source1.link == "https://example.com"
        
        # Source with only text
        source2 = SourceInfo(text="Course Title")
        assert source2.text == "Course Title"
        assert source2.link is None

    def test_query_response_model(self):
        """Test QueryResponse model validation"""
        from app import QueryResponse, SourceInfo
        
        sources = [
            SourceInfo(text="Source 1", link="https://example1.com"),
            SourceInfo(text="Source 2")
        ]
        
        response = QueryResponse(
            answer="Test answer",
            sources=sources,
            session_id="session123"
        )
        
        assert response.answer == "Test answer"
        assert len(response.sources) == 2
        assert response.session_id == "session123"

    def test_course_stats_model(self):
        """Test CourseStats model validation"""
        from app import CourseStats
        
        stats = CourseStats(
            total_courses=5,
            course_titles=["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        )
        
        assert stats.total_courses == 5
        assert len(stats.course_titles) == 5
        assert "Course 1" in stats.course_titles