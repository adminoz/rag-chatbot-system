import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test RAGSystem integration"""
    
    @pytest.fixture
    def rag_system(self, test_config):
        """Create RAG system with test configuration"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response")]
            mock_response.stop_reason = "end_turn"
            mock_anthropic.return_value.messages.create.return_value = mock_response
            
            return RAGSystem(test_config)
    
    def test_initialization(self, rag_system, test_config):
        """Test RAG system initialization"""
        assert rag_system.config == test_config
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None
        
        # Verify tools are registered
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [d["name"] for d in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    def test_add_course_document_success(self, rag_system):
        """Test adding a single course document successfully"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""Course Title: Test Course
Course Instructor: Test Instructor
Course Link: https://example.com/course

Lesson 1: Introduction
This is lesson 1 content.

Lesson 2: Advanced Topics  
This is lesson 2 content.""")
            temp_file = f.name
        
        try:
            course, chunk_count = rag_system.add_course_document(temp_file)
            
            assert course is not None
            assert course.title == "Test Course"
            assert course.instructor == "Test Instructor"
            assert chunk_count > 0
            
            # Verify data was added to vector store
            titles = rag_system.vector_store.get_existing_course_titles()
            assert "Test Course" in titles
            
        finally:
            os.unlink(temp_file)
    
    def test_add_course_document_invalid_file(self, rag_system):
        """Test adding non-existent course document"""
        course, chunk_count = rag_system.add_course_document("/nonexistent/file.txt")
        
        assert course is None
        assert chunk_count == 0
    
    def test_add_course_folder_success(self, rag_system):
        """Test adding course documents from folder"""
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file1 = os.path.join(temp_dir, "course1.txt")
            with open(test_file1, 'w') as f:
                f.write("""Course Title: Course 1
Course Instructor: Instructor 1
Course Link: https://example.com/course1

Lesson 1: Introduction
Content for course 1.""")
            
            test_file2 = os.path.join(temp_dir, "course2.txt")
            with open(test_file2, 'w') as f:
                f.write("""Course Title: Course 2
Course Instructor: Instructor 2
Course Link: https://example.com/course2

Lesson 1: Introduction  
Content for course 2.""")
            
            courses, chunks = rag_system.add_course_folder(temp_dir, clear_existing=True)
            
            assert courses == 2
            assert chunks > 0
            
            # Verify courses were added
            titles = rag_system.vector_store.get_existing_course_titles()
            assert "Course 1" in titles
            assert "Course 2" in titles
    
    def test_add_course_folder_nonexistent(self, rag_system):
        """Test adding from non-existent folder"""
        courses, chunks = rag_system.add_course_folder("/nonexistent/folder")
        
        assert courses == 0
        assert chunks == 0
    
    def test_add_course_folder_skip_existing(self, rag_system):
        """Test that existing courses are skipped"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "course.txt")
            with open(test_file, 'w') as f:
                f.write("""Course Title: Existing Course
Course Instructor: Test Instructor
Course Link: https://example.com/course

Lesson 1: Introduction
Content here.""")
            
            # Add once
            courses1, chunks1 = rag_system.add_course_folder(temp_dir, clear_existing=True)
            
            # Add again without clearing
            courses2, chunks2 = rag_system.add_course_folder(temp_dir, clear_existing=False)
            
            assert courses1 == 1
            assert courses2 == 0  # Should skip existing course
    
    def test_query_without_session(self, rag_system):
        """Test query processing without session ID"""
        query = "What is machine learning?"
        
        response, sources = rag_system.query(query)
        
        assert isinstance(response, str)
        assert isinstance(sources, list)
        assert len(response) > 0
    
    def test_query_with_session(self, rag_system):
        """Test query processing with session ID"""
        query = "What is machine learning?"
        session_id = "test_session"
        
        response, sources = rag_system.query(query, session_id=session_id)
        
        assert isinstance(response, str)
        assert isinstance(sources, list)
        
        # Verify conversation history was updated
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert query in history
    
    def test_query_with_tools(self, test_config):
        """Test query processing that triggers tool use"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            # Setup mock for tool use response
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock tool use response
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "search_course_content"
            mock_tool_content.input = {"query": "machine learning"}
            mock_tool_content.id = "tool_123"
            
            tool_use_response = Mock()
            tool_use_response.content = [mock_tool_content]
            tool_use_response.stop_reason = "tool_use"
            
            # Mock final response
            final_response = Mock()
            final_response.content = [Mock(text="Machine learning is...")]
            
            mock_client.messages.create.side_effect = [tool_use_response, final_response]
            
            # Create RAG system with populated data
            rag_system = RAGSystem(test_config)
            
            # Add some test data
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = os.path.join(temp_dir, "ml_course.txt")
                with open(test_file, 'w') as f:
                    f.write("""Course Title: Machine Learning Basics
Course Instructor: Dr. Smith
Course Link: https://example.com/ml-basics

Lesson 1: Introduction
Machine learning is a powerful technology.""")
                
                rag_system.add_course_folder(temp_dir, clear_existing=True)
            
            # Query the system
            response, sources = rag_system.query("What is machine learning?")
            
            assert response == "Machine learning is..."
            assert isinstance(sources, list)
    
    def test_query_prompt_formatting(self, rag_system):
        """Test that query is properly formatted as prompt"""
        query = "What is machine learning?"
        
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "Test response"
            
            rag_system.query(query)
            
            # Verify the prompt was properly formatted
            call_args = mock_generate.call_args
            prompt = call_args[1]['query']  # keyword argument
            assert f"Answer this question about course materials: {query}" == prompt
    
    def test_query_conversation_history_handling(self, rag_system):
        """Test conversation history is passed to AI generator"""
        query = "What is machine learning?"
        session_id = "test_session"
        
        # Add some history first
        rag_system.session_manager.add_exchange(
            session_id,
            "Previous question",
            "Previous answer"
        )
        
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "Test response"
            
            rag_system.query(query, session_id=session_id)
            
            # Verify history was passed
            call_args = mock_generate.call_args
            history = call_args[1]['conversation_history']
            assert history is not None
            assert "Previous question" in history
    
    def test_query_tools_integration(self, rag_system):
        """Test that tools are properly passed to AI generator"""
        query = "What is machine learning?"
        
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "Test response"
            
            rag_system.query(query)
            
            # Verify tools were passed
            call_args = mock_generate.call_args
            tools = call_args[1]['tools']
            tool_manager = call_args[1]['tool_manager']
            
            assert tools is not None
            assert tool_manager is rag_system.tool_manager
            
            # Verify tool definitions
            tool_names = [tool["name"] for tool in tools]
            assert "search_course_content" in tool_names
    
    def test_query_sources_handling(self, rag_system):
        """Test that sources are properly retrieved and reset"""
        query = "What is machine learning?"
        
        # Mock tool manager to return sources
        mock_sources = [{"text": "Course 1", "link": "http://example.com"}]
        rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query(query)
        
        assert sources == mock_sources
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_query_session_creation(self, rag_system):
        """Test that conversation history is updated after query"""
        query = "What is machine learning?"
        session_id = "test_session"
        expected_response = "Machine learning is..."
        
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = expected_response
            
            response, sources = rag_system.query(query, session_id=session_id)
            
            # Verify session was updated
            history = rag_system.session_manager.get_conversation_history(session_id)
            assert query in history
            assert expected_response in history
    
    def test_get_course_analytics(self, rag_system):
        """Test course analytics retrieval"""
        # Add some test data
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "course.txt")
            with open(test_file, 'w') as f:
                f.write("""Course Title: Analytics Test Course
Course Instructor: Test Instructor
Course Link: https://example.com/analytics-course

Lesson 1: Introduction
Test content.""")
            
            rag_system.add_course_folder(temp_dir, clear_existing=True)
        
        analytics = rag_system.get_course_analytics()
        
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert analytics["total_courses"] >= 1
        assert "Analytics Test Course" in analytics["course_titles"]
    
    def test_error_handling_in_query(self, rag_system):
        """Test error handling during query processing"""
        query = "What is machine learning?"
        
        # Mock AI generator to raise exception
        with patch.object(rag_system.ai_generator, 'generate_response', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                rag_system.query(query)
    
    def test_component_integration(self, rag_system):
        """Test that all components are properly integrated"""
        # Verify all components are initialized and connected
        assert rag_system.search_tool.store == rag_system.vector_store
        assert rag_system.outline_tool.store == rag_system.vector_store
        
        # Verify tools are registered in tool manager
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools
        
        # Verify the registered tools are the same instances
        assert rag_system.tool_manager.tools["search_course_content"] == rag_system.search_tool