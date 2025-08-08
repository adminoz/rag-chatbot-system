import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""
    
    def test_initialization(self, test_config):
        """Test AIGenerator initialization"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            assert generator.model == test_config.ANTHROPIC_MODEL
            assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
            mock_anthropic.assert_called_once_with(api_key=test_config.ANTHROPIC_API_KEY)
    
    def test_generate_response_without_tools(self, ai_generator):
        """Test generating response without tools"""
        query = "What is machine learning?"
        
        response = ai_generator.generate_response(query)
        
        assert isinstance(response, str)
        assert response == "Test response"  # From our mock
    
    def test_generate_response_with_conversation_history(self, ai_generator):
        """Test generating response with conversation history"""
        query = "What is machine learning?"
        history = "Previous conversation content"
        
        response = ai_generator.generate_response(query, conversation_history=history)
        
        assert isinstance(response, str)
        # Verify that system prompt includes history
        call_args = ai_generator.client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content
    
    def test_generate_response_with_tools_no_tool_use(self, ai_generator):
        """Test generating response with tools available but not used"""
        query = "What is machine learning?"
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        mock_tool_manager = Mock()
        
        response = ai_generator.generate_response(query, tools=tools, tool_manager=mock_tool_manager)
        
        assert isinstance(response, str)
        assert response == "Test response"
        
        # Verify tools were included in API call
        call_args = ai_generator.client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_single_tool_use(self, test_config):
        """Test generating response with single tool use round"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            # Setup mock client
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock initial response with tool use
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "search_course_content"
            mock_tool_content.input = {"query": "machine learning"}
            mock_tool_content.id = "tool_123"
            
            initial_response = Mock()
            initial_response.content = [mock_tool_content]
            initial_response.stop_reason = "tool_use"
            
            # Mock final response after tool execution
            final_response = Mock()
            final_response.content = [Mock(text="Final response with tool results")]
            
            # Configure mock to return different responses
            mock_client.messages.create.side_effect = [initial_response, final_response]
            
            # Setup tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool execution result"
            
            # Create AI generator and test
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                    "input_schema": {"type": "object", "properties": {}}
                }
            ]
            
            response = generator.generate_response(
                "What is machine learning?",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "Final response with tool results"
            
            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="machine learning"
            )
            
            # Verify two API calls were made
            assert mock_client.messages.create.call_count == 2
    
    def test_generate_response_with_sequential_tool_use(self, test_config):
        """Test generating response with sequential tool calls (2 rounds)"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            # Setup mock client
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock first round - tool use
            mock_tool1 = Mock()
            mock_tool1.type = "tool_use"
            mock_tool1.name = "get_course_outline"
            mock_tool1.input = {"course_id": "course1"}
            mock_tool1.id = "tool1"
            
            first_response = Mock()
            first_response.content = [mock_tool1]
            first_response.stop_reason = "tool_use"
            
            # Mock second round - another tool use
            mock_tool2 = Mock()
            mock_tool2.type = "tool_use"
            mock_tool2.name = "search_course_content"
            mock_tool2.input = {"query": "lesson 4 content"}
            mock_tool2.id = "tool2"
            
            second_response = Mock()
            second_response.content = [mock_tool2]
            second_response.stop_reason = "tool_use"
            
            # Mock final response
            final_response = Mock()
            final_response.content = [Mock(text="Final synthesized response")]
            
            # Configure mock responses
            mock_client.messages.create.side_effect = [
                first_response, second_response, final_response
            ]
            
            # Setup tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = [
                "Course outline result", "Search result"
            ]
            
            # Create AI generator and test
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            tools = [
                {"name": "get_course_outline", "description": "Get course outline"},
                {"name": "search_course_content", "description": "Search content"}
            ]
            
            response = generator.generate_response(
                "Find lesson 4 of course X and search for similar content",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "Final synthesized response"
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_id="course1")
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 4 content")
            
            # Verify three API calls were made (2 tool rounds + 1 final)
            assert mock_client.messages.create.call_count == 3
    
    def test_generate_response_max_rounds_reached(self, test_config):
        """Test that generation stops after max rounds (2)"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock tool responses for both rounds
            mock_tool = Mock()
            mock_tool.type = "tool_use"
            mock_tool.name = "search_course_content"
            mock_tool.input = {"query": "test"}
            mock_tool.id = "tool_id"
            
            tool_response = Mock()
            tool_response.content = [mock_tool]
            tool_response.stop_reason = "tool_use"
            
            final_response = Mock()
            final_response.content = [Mock(text="Final response after 2 rounds")]
            
            # Configure to return tool responses for both rounds, then final
            mock_client.messages.create.side_effect = [
                tool_response, tool_response, final_response
            ]
            
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"
            
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = [{"name": "search_course_content", "description": "Search"}]
            
            response = generator.generate_response(
                "Test query", tools=tools, tool_manager=mock_tool_manager
            )
            
            assert response == "Final response after 2 rounds"
            # Should execute tools exactly 2 times (max rounds)
            assert mock_tool_manager.execute_tool.call_count == 2
            # Should make 3 API calls (2 tool rounds + 1 final)
            assert mock_client.messages.create.call_count == 3
    
    def test_generate_response_tool_execution_failure(self, test_config):
        """Test handling of tool execution failures"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock tool use response
            mock_tool = Mock()
            mock_tool.type = "tool_use"
            mock_tool.name = "search_course_content"
            mock_tool.input = {"query": "test"}
            mock_tool.id = "tool_id"
            
            tool_response = Mock()
            tool_response.content = [mock_tool]
            tool_response.stop_reason = "tool_use"
            
            final_response = Mock()
            final_response.content = [Mock(text="Response with error handling")]
            
            mock_client.messages.create.side_effect = [tool_response, final_response]
            
            # Setup tool manager to raise exception
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
            
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = [{"name": "search_course_content", "description": "Search"}]
            
            response = generator.generate_response(
                "Test query", tools=tools, tool_manager=mock_tool_manager
            )
            
            assert response == "Response with error handling"
            # Should attempt tool execution once
            assert mock_tool_manager.execute_tool.call_count == 1
            # Should make 2 API calls (1 tool round + 1 final after error)
            assert mock_client.messages.create.call_count == 2
    
    def test_generate_response_stops_when_no_tools_requested(self, test_config):
        """Test that generation stops when Claude doesn't request tools"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock response without tool use
            text_response = Mock()
            text_response.content = [Mock(text="Direct answer without tools")]
            text_response.stop_reason = "end_turn"
            
            mock_client.messages.create.return_value = text_response
            
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = [{"name": "search_course_content", "description": "Search"}]
            
            response = generator.generate_response(
                "What is 2+2?", tools=tools, tool_manager=Mock()
            )
            
            assert response == "Direct answer without tools"
            # Should make only one API call
            assert mock_client.messages.create.call_count == 1
    
    def test_system_prompt_format(self, ai_generator):
        """Test that system prompt is properly formatted"""
        # The system prompt should contain instructions for tool use
        assert "search_course_content" in ai_generator.SYSTEM_PROMPT
        assert "get_course_outline" in ai_generator.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in ai_generator.SYSTEM_PROMPT
        assert "Response Protocol" in ai_generator.SYSTEM_PROMPT
        assert "Sequential tool usage" in ai_generator.SYSTEM_PROMPT
        assert "up to 2 rounds" in ai_generator.SYSTEM_PROMPT
        assert "Multi-round scenarios" in ai_generator.SYSTEM_PROMPT
    
    def test_api_error_handling(self, test_config):
        """Test API error handling"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock API error
            mock_client.messages.create.side_effect = Exception("API Error")
            
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            with pytest.raises(Exception, match="API Error"):
                generator.generate_response("test query")
    
    def test_base_params_configuration(self, ai_generator):
        """Test that base parameters are correctly configured"""
        expected_params = {
            "model": ai_generator.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        assert ai_generator.base_params == expected_params
    
    def test_conversation_history_integration(self, ai_generator):
        """Test that conversation history is properly integrated into system prompt"""
        query = "What is machine learning?"
        history = "User: Hello\nAssistant: Hi there!"
        
        ai_generator.generate_response(query, conversation_history=history)
        
        # Verify the API call
        call_args = ai_generator.client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        
        assert ai_generator.SYSTEM_PROMPT in system_content
        assert "Previous conversation:" in system_content
        assert history in system_content
    
    def test_no_conversation_history(self, ai_generator):
        """Test response generation without conversation history"""
        query = "What is machine learning?"
        
        ai_generator.generate_response(query)
        
        # Verify the API call
        call_args = ai_generator.client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        
        assert system_content == ai_generator.SYSTEM_PROMPT
        assert "Previous conversation:" not in system_content
    
    def test_api_logging_functionality(self, test_config, caplog):
        """Test that API calls are properly logged with timestamps and duration"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock response
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response")]
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            # Capture logs
            with caplog.at_level(logging.INFO):
                response = generator.generate_response("Test query")
            
            assert response == "Test response"
            
            # Verify logging occurred
            log_records = caplog.records
            assert len(log_records) == 2  # Start and completion logs
            
            # Check start log
            start_log = log_records[0]
            assert "Anthropic API call started" in start_log.message
            assert "Initial" in start_log.message
            assert "Timestamp:" in start_log.message
            assert f"Model: {test_config.ANTHROPIC_MODEL}" in start_log.message
            assert "Tools: disabled" in start_log.message
            
            # Check completion log
            completion_log = log_records[1]
            assert "Anthropic API call completed" in completion_log.message
            assert "Initial" in completion_log.message
            assert "Duration:" in completion_log.message
            assert "Stop reason: end_turn" in completion_log.message
            assert "Content blocks: 1" in completion_log.message
    
    def test_api_logging_with_tools(self, test_config, caplog):
        """Test logging for API calls with tools enabled"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock tool response
            mock_tool = Mock()
            mock_tool.type = "tool_use"
            mock_tool.name = "search_course_content"
            mock_tool.input = {"query": "test"}
            mock_tool.id = "tool_123"
            
            tool_response = Mock()
            tool_response.content = [mock_tool]
            tool_response.stop_reason = "tool_use"
            
            final_response = Mock()
            final_response.content = [Mock(text="Final response")]
            final_response.stop_reason = "end_turn"
            
            mock_client.messages.create.side_effect = [tool_response, final_response]
            
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"
            
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = [{"name": "search_course_content", "description": "Search"}]
            
            with caplog.at_level(logging.INFO):
                response = generator.generate_response(
                    "Test query", tools=tools, tool_manager=mock_tool_manager
                )
            
            assert response == "Final response"
            
            # Should have 4 log entries (2 API calls x 2 logs each)
            log_records = caplog.records
            assert len(log_records) == 4
            
            # Check first round logs
            assert "Initial" in log_records[0].message
            assert "Tools: enabled" in log_records[0].message
            assert "Stop reason: tool_use" in log_records[1].message
            
            # Check second round logs  
            assert "Round 2" in log_records[2].message
            assert "Tools: enabled" in log_records[2].message
            assert "Stop reason: end_turn" in log_records[3].message
    
    def test_api_logging_on_error(self, test_config, caplog):
        """Test logging when API call fails"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock API error
            mock_client.messages.create.side_effect = Exception("API Error")
            
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            with caplog.at_level(logging.INFO):
                with pytest.raises(Exception, match="API Error"):
                    generator.generate_response("Test query")
            
            # Should have 2 log entries (start and error)
            log_records = caplog.records
            assert len(log_records) == 2
            
            # Check start log
            assert "Anthropic API call started" in log_records[0].message
            assert log_records[0].levelname == "INFO"
            
            # Check error log
            assert "Anthropic API call failed" in log_records[1].message
            assert "Error: API Error" in log_records[1].message
            assert log_records[1].levelname == "ERROR"