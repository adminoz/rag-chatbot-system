import anthropic
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
- **search_course_content**: Search within course materials for specific content, topics, or detailed educational information
- **get_course_outline**: Get complete course outlines including course title, course link, and full lesson listings

Tool Usage Guidelines:
- **Course outline requests**: Use get_course_outline for questions about course structure, lesson lists, or course overviews
- **Content/topic questions**: Use search_course_content for specific course content, explanations, or detailed materials
- **Sequential tool usage**: You may use multiple tool calls across up to 2 rounds to gather comprehensive information for complex queries
- **Multi-round scenarios**: Use sequential calls for comparisons, multi-part questions, or when information from different courses/lessons is needed
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline queries**: Use get_course_outline to provide course title, course link, and complete lesson breakdown
- **Course content queries**: Use search_course_content first, then answer
- **Complex queries**: Break down into multiple tool calls if needed (e.g., get outline first, then search for specific content)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def _call_anthropic_api(self, api_params: Dict[str, Any], round_info: str = "") -> Any:
        """
        Make an API call to Anthropic with logging.
        
        Args:
            api_params: Parameters for the API call
            round_info: Additional context for logging (e.g., "Round 1", "Final")
            
        Returns:
            API response object
        """
        timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        # Log the API call start
        self.logger.info(
            f"Anthropic API call started - {round_info} - "
            f"Timestamp: {timestamp} - "
            f"Model: {api_params.get('model', 'unknown')} - "
            f"Tools: {'enabled' if api_params.get('tools') else 'disabled'}"
        )
        
        try:
            response = self.client.messages.create(**api_params)
            duration = time.time() - start_time
            
            # Log successful completion
            self.logger.info(
                f"Anthropic API call completed - {round_info} - "
                f"Duration: {duration:.3f}s - "
                f"Stop reason: {response.stop_reason} - "
                f"Content blocks: {len(response.content)}"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            self.logger.error(
                f"Anthropic API call failed - {round_info} - "
                f"Duration: {duration:.3f}s - "
                f"Error: {str(e)}"
            )
            
            raise e
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential tool calling rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        
        # Sequential tool execution loop (max 2 rounds)
        max_rounds = 2
        round_count = 0
        
        while round_count < max_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            
            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            try:
                # Get response from Claude with logging
                round_info = f"Round {round_count + 1}" if round_count > 0 else "Initial"
                response = self._call_anthropic_api(api_params, round_info)
                
                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use" and tool_manager:
                    round_count += 1
                    
                    # Add AI's response to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    
                    # Execute tools and collect results
                    tool_results = []
                    tool_execution_failed = False
                    
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            try:
                                tool_result = tool_manager.execute_tool(
                                    content_block.name, 
                                    **content_block.input
                                )
                                
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": content_block.id,
                                    "content": tool_result
                                })
                            except Exception as e:
                                # Handle tool execution failure
                                tool_execution_failed = True
                                tool_results.append({
                                    "type": "tool_result", 
                                    "tool_use_id": content_block.id,
                                    "content": f"Tool execution failed: {str(e)}"
                                })
                    
                    # Add tool results to conversation
                    if tool_results:
                        messages.append({"role": "user", "content": tool_results})
                    
                    # Stop if tool execution failed
                    if tool_execution_failed:
                        break
                        
                    # Continue to next round
                    continue
                else:
                    # No tool use - return direct response
                    return response.content[0].text
                    
            except Exception as e:
                # Handle API errors
                if round_count == 0:
                    raise e
                else:
                    # Return partial result if we have some progress
                    return f"Error in round {round_count + 1}: {str(e)}"
        
        # Final call to get synthesized response after tool rounds
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        try:
            final_response = self._call_anthropic_api(final_params, "Final")
            return final_response.content[0].text
        except Exception as e:
            return f"Error generating final response: {str(e)}"
