import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""
    
    def test_get_tool_definition(self, course_search_tool):
        """Test tool definition format"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert "properties" in definition["input_schema"]
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_basic_query(self, course_search_tool):
        """Test basic query execution"""
        result = course_search_tool.execute("machine learning")
        
        # Should return formatted results
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should contain course information
        if "No relevant content found" not in result:
            assert "[" in result  # Course headers
            assert "]" in result
    
    def test_execute_with_course_filter(self, course_search_tool):
        """Test query with course name filter"""
        result = course_search_tool.execute("machine learning", course_name="Introduction to Machine Learning")
        
        assert isinstance(result, str)
        # Should find content since we have ML course data
        if "No relevant content found" not in result:
            assert "Introduction to Machine Learning" in result
    
    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test query with lesson number filter"""
        result = course_search_tool.execute("linear regression", lesson_number=2)
        
        assert isinstance(result, str)
        if "No relevant content found" not in result:
            assert "Lesson 2" in result
    
    def test_execute_with_both_filters(self, course_search_tool):
        """Test query with both course and lesson filters"""
        result = course_search_tool.execute(
            "machine learning",
            course_name="Introduction to Machine Learning", 
            lesson_number=1
        )
        
        assert isinstance(result, str)
        if "No relevant content found" not in result:
            assert "Introduction to Machine Learning" in result
            assert "Lesson 1" in result
    
    def test_execute_no_results(self, course_search_tool):
        """Test query with low relevance terms"""
        result = course_search_tool.execute("quantum physics")
        
        assert isinstance(result, str)
        assert len(result) > 0  # Should return some results even if not highly relevant
    
    def test_execute_invalid_course(self, course_search_tool):
        """Test query with non-existent course name falls back to general search"""
        result = course_search_tool.execute("test query", course_name="Nonexistent Course")
        
        assert isinstance(result, str)
        assert len(result) > 0  # Should fall back to general search
    
    def test_execute_vector_store_error(self, vector_store):
        """Test handling of vector store errors"""
        # Create tool with mocked vector store that raises exception
        with patch.object(vector_store, 'search', side_effect=Exception("Database error")):
            tool = CourseSearchTool(vector_store)
            
            # Should raise the exception rather than handle it silently
            with pytest.raises(Exception, match="Database error"):
                tool.execute("test query")
    
    def test_format_results_with_lesson_links(self, vector_store):
        """Test result formatting with lesson links"""
        tool = CourseSearchTool(vector_store)
        
        # Create mock search results
        results = SearchResults(
            documents=["Test content 1", "Test content 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            lesson_links=["http://example.com/lesson1", "http://example.com/lesson2"]
        )
        
        formatted = tool._format_results(results)
        
        assert isinstance(formatted, str)
        assert "[Test Course - Lesson 1]" in formatted
        assert "[Test Course - Lesson 2]" in formatted
        assert "Test content 1" in formatted
        assert "Test content 2" in formatted
        
        # Check that sources are tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["link"] == "http://example.com/lesson1"
    
    def test_format_results_without_lesson_info(self, vector_store):
        """Test result formatting without lesson information"""
        tool = CourseSearchTool(vector_store)
        
        results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1],
            lesson_links=[None]
        )
        
        formatted = tool._format_results(results)
        
        assert "[Test Course]" in formatted
        assert "Test content" in formatted
        
        # Check source tracking
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""
    
    def test_get_tool_definition(self, populated_vector_store):
        """Test tool definition format"""
        tool = CourseOutlineTool(populated_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert "course_title" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_title"]
    
    def test_execute_valid_course(self, populated_vector_store):
        """Test outline retrieval for valid course"""
        tool = CourseOutlineTool(populated_vector_store)
        result = tool.execute("Introduction to Machine Learning")
        
        assert isinstance(result, str)
        assert "Course:" in result
        assert "Introduction to Machine Learning" in result
        assert "Instructor:" in result
        assert "Lessons:" in result
    
    def test_execute_partial_course_name(self, populated_vector_store):
        """Test outline retrieval with partial course name"""
        tool = CourseOutlineTool(populated_vector_store)
        result = tool.execute("Machine Learning")  # Partial name
        
        assert isinstance(result, str)
        # Should still find the course due to semantic matching
        if "No course found" not in result:
            assert "Introduction to Machine Learning" in result
    
    def test_execute_invalid_course(self, populated_vector_store):
        """Test outline retrieval for non-existent course returns best match"""
        tool = CourseOutlineTool(populated_vector_store)
        result = tool.execute("Nonexistent Course")
        
        assert isinstance(result, str)
        assert len(result) > 0  # Should return the closest matching course outline


class TestToolManager:
    """Test ToolManager functionality"""
    
    def test_register_tool(self, course_search_tool):
        """Test tool registration"""
        manager = ToolManager()
        manager.register_tool(course_search_tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == course_search_tool
    
    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions"""
        definitions = tool_manager.get_tool_definitions()
        
        assert isinstance(definitions, list)
        assert len(definitions) > 0
        
        # Should include our search tool
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
    
    def test_execute_tool_success(self, tool_manager):
        """Test successful tool execution"""
        result = tool_manager.execute_tool("search_course_content", query="machine learning")
        
        assert isinstance(result, str)
        # Should either return results or "no content found" message
        assert len(result) > 0
    
    def test_execute_tool_not_found(self, tool_manager):
        """Test execution of non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert isinstance(result, str)
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, tool_manager):
        """Test source tracking"""
        # Execute a search to generate sources
        tool_manager.execute_tool("search_course_content", query="machine learning")
        
        sources = tool_manager.get_last_sources()
        assert isinstance(sources, list)
        # Sources might be empty if no results found, but should be a list
    
    def test_reset_sources(self, tool_manager):
        """Test source resetting"""
        # Execute a search to generate sources
        tool_manager.execute_tool("search_course_content", query="machine learning")
        
        # Reset sources
        tool_manager.reset_sources()
        
        sources = tool_manager.get_last_sources()
        assert sources == []
    
    def test_register_invalid_tool(self):
        """Test registering tool without name"""
        manager = ToolManager()
        
        # Mock tool with invalid definition
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "test"}  # No name
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)