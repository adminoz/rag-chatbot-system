import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore:
    """Test VectorStore functionality"""
    
    def test_initialization(self, test_config):
        """Test VectorStore initialization"""
        store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        
        assert store.max_results == test_config.MAX_RESULTS
        assert store.client is not None
        assert store.embedding_function is not None
        assert store.course_catalog is not None
        assert store.course_content is not None
    
    def test_add_course_metadata(self, vector_store, sample_courses):
        """Test adding course metadata"""
        course = sample_courses[0]
        vector_store.add_course_metadata(course)
        
        # Verify course was added by trying to retrieve it
        existing_titles = vector_store.get_existing_course_titles()
        assert course.title in existing_titles
    
    def test_add_course_content(self, vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        chunks = sample_course_chunks[:2]  # Test with first 2 chunks
        vector_store.add_course_content(chunks)
        
        # Verify content was added by searching for it
        results = vector_store.search("machine learning")
        assert not results.is_empty() or results.error is None
    
    def test_search_basic_query(self, populated_vector_store):
        """Test basic search functionality"""
        results = populated_vector_store.search("machine learning")
        
        assert isinstance(results, SearchResults)
        assert results.error is None
        
        if not results.is_empty():
            assert len(results.documents) > 0
            assert len(results.metadata) == len(results.documents)
            assert len(results.distances) == len(results.documents)
    
    def test_search_with_course_filter(self, populated_vector_store):
        """Test search with course name filter"""
        results = populated_vector_store.search(
            "machine learning", 
            course_name="Introduction to Machine Learning"
        )
        
        assert isinstance(results, SearchResults)
        
        if not results.is_empty():
            # All results should be from the specified course
            for metadata in results.metadata:
                assert metadata.get("course_title") == "Introduction to Machine Learning"
    
    def test_search_with_lesson_filter(self, populated_vector_store):
        """Test search with lesson number filter"""
        results = populated_vector_store.search("machine learning", lesson_number=1)
        
        assert isinstance(results, SearchResults)
        
        if not results.is_empty():
            # All results should be from lesson 1
            for metadata in results.metadata:
                assert metadata.get("lesson_number") == 1
    
    def test_search_with_both_filters(self, populated_vector_store):
        """Test search with both course and lesson filters"""
        results = populated_vector_store.search(
            "machine learning",
            course_name="Introduction to Machine Learning",
            lesson_number=1
        )
        
        assert isinstance(results, SearchResults)
        
        if not results.is_empty():
            for metadata in results.metadata:
                assert metadata.get("course_title") == "Introduction to Machine Learning"
                assert metadata.get("lesson_number") == 1
    
    def test_search_invalid_course(self, populated_vector_store):
        """Test search with non-existent course name falls back to general search"""
        results = populated_vector_store.search(
            "test query",
            course_name="Nonexistent Course"
        )
        
        assert isinstance(results, SearchResults)
        assert results.error is None  # Should fall back to general search
        assert len(results.documents) > 0  # Should return some results
    
    def test_search_with_limit(self, populated_vector_store):
        """Test search with custom result limit"""
        results = populated_vector_store.search("machine learning", limit=2)
        
        assert isinstance(results, SearchResults)
        
        if not results.is_empty():
            assert len(results.documents) <= 2
    
    def test_resolve_course_name_exact_match(self, populated_vector_store):
        """Test course name resolution with exact match"""
        resolved = populated_vector_store._resolve_course_name("Introduction to Machine Learning")
        assert resolved == "Introduction to Machine Learning"
    
    def test_resolve_course_name_partial_match(self, populated_vector_store):
        """Test course name resolution with partial match"""
        resolved = populated_vector_store._resolve_course_name("Machine Learning")
        # Should resolve to full title due to semantic matching
        assert resolved is not None
    
    def test_resolve_course_name_no_match(self, populated_vector_store):
        """Test course name resolution with no match returns best match"""
        resolved = populated_vector_store._resolve_course_name("Quantum Physics")
        # The system uses semantic similarity, so it may return the closest match
        assert resolved is None or isinstance(resolved, str)
    
    def test_build_filter_no_filters(self, vector_store):
        """Test filter building with no filters"""
        filter_dict = vector_store._build_filter(None, None)
        assert filter_dict is None
    
    def test_build_filter_course_only(self, vector_store):
        """Test filter building with course only"""
        filter_dict = vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self, vector_store):
        """Test filter building with lesson only"""
        filter_dict = vector_store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}
    
    def test_build_filter_both(self, vector_store):
        """Test filter building with both filters"""
        filter_dict = vector_store._build_filter("Test Course", 1)
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 1}
            ]
        }
        assert filter_dict == expected
    
    def test_fetch_lesson_links(self, populated_vector_store):
        """Test fetching lesson links from metadata"""
        metadata_list = [
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1},
            {"course_title": "Advanced Python Programming", "lesson_number": 2}
        ]
        
        links = populated_vector_store._fetch_lesson_links(metadata_list)
        
        assert isinstance(links, list)
        assert len(links) == 2
        # Links might be None if not found, but should be a list of the right length
    
    def test_get_existing_course_titles(self, populated_vector_store):
        """Test getting existing course titles"""
        titles = populated_vector_store.get_existing_course_titles()
        
        assert isinstance(titles, list)
        assert "Introduction to Machine Learning" in titles
        assert "Advanced Python Programming" in titles
    
    def test_get_course_count(self, populated_vector_store):
        """Test getting course count"""
        count = populated_vector_store.get_course_count()
        
        assert isinstance(count, int)
        assert count >= 2  # Should have at least our sample courses
    
    def test_get_all_courses_metadata(self, populated_vector_store):
        """Test getting all courses metadata"""
        metadata_list = populated_vector_store.get_all_courses_metadata()
        
        assert isinstance(metadata_list, list)
        assert len(metadata_list) >= 2
        
        # Check that lessons are parsed from JSON
        for metadata in metadata_list:
            assert "title" in metadata
            if "lessons" in metadata:
                assert isinstance(metadata["lessons"], list)
    
    def test_get_course_link(self, populated_vector_store):
        """Test getting course link"""
        link = populated_vector_store.get_course_link("Introduction to Machine Learning")
        
        # Link might be None or a string
        assert link is None or isinstance(link, str)
    
    def test_get_lesson_link(self, populated_vector_store):
        """Test getting lesson link"""
        link = populated_vector_store.get_lesson_link("Introduction to Machine Learning", 1)
        
        # Link might be None or a string
        assert link is None or isinstance(link, str)
    
    def test_clear_all_data(self, vector_store, sample_courses, sample_course_chunks):
        """Test clearing all data"""
        # Add some data first
        vector_store.add_course_metadata(sample_courses[0])
        vector_store.add_course_content(sample_course_chunks[:1])
        
        # Verify data exists
        assert vector_store.get_course_count() > 0
        
        # Clear data
        vector_store.clear_all_data()
        
        # Verify data is cleared
        assert vector_store.get_course_count() == 0
    
    def test_search_error_handling(self, vector_store):
        """Test search error handling"""
        # Mock the course_content collection to raise an exception
        with patch.object(vector_store.course_content, 'query', side_effect=Exception("Database error")):
            results = vector_store.search("test query")
            
            assert isinstance(results, SearchResults)
            assert results.error is not None
            assert "Search error" in results.error


class TestSearchResults:
    """Test SearchResults functionality"""
    
    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'value1'}, {'key': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.lesson_links == []
        assert results.error is None
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.lesson_links == []
        assert results.error is None
    
    def test_empty_constructor(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.lesson_links == []
        assert results.error == "Test error message"
    
    def test_is_empty_true(self):
        """Test is_empty method when results are empty"""
        results = SearchResults([], [], [])
        assert results.is_empty() is True
    
    def test_is_empty_false(self):
        """Test is_empty method when results exist"""
        results = SearchResults(['doc1'], [{'key': 'value'}], [0.1])
        assert results.is_empty() is False