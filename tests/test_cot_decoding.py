"""
Unit tests for CoT decoding modules.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGSMTask:
    """Test suite for GSMTask class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from cot_dataset.cot_decoding.task import GSMTask
        self.GSMTask = GSMTask
    
    def test_initialization_instruct_format(self):
        """Test GSMTask initialization with instruct format."""
        task = self.GSMTask(encode_format='instruct')
        assert task.encode_format == 'instruct'
    
    def test_initialization_qa_format(self):
        """Test GSMTask initialization with qa format."""
        task = self.GSMTask(encode_format='qa')
        assert task.encode_format == 'qa'
    
    def test_initialization_invalid_format_raises_error(self):
        """Test that invalid format raises assertion error."""
        with pytest.raises(AssertionError):
            self.GSMTask(encode_format='invalid')
    
    def test_encode_prompt_instruct(self):
        """Test prompt encoding with instruct format."""
        task = self.GSMTask(encode_format='instruct')
        example = {'question': 'What is 2 + 2?'}
        
        result = task.encode_prompt(example)
        
        assert result == '[INST]What is 2 + 2?[/INST]'
    
    def test_encode_prompt_qa(self):
        """Test prompt encoding with qa format."""
        task = self.GSMTask(encode_format='qa')
        example = {'question': 'What is 2 + 2?'}
        
        result = task.encode_prompt(example)
        
        assert result == 'Q: What is 2 + 2?\nA:'
    
    def test_extract_gt_answer_valid(self):
        """Test extraction of ground truth answer."""
        task = self.GSMTask(encode_format='instruct')
        completion = "The answer is 42\n#### 42"
        
        result = task.extract_gt_answer(completion)
        
        assert result == "42"
    
    def test_extract_gt_answer_with_comma(self):
        """Test extraction of answer with comma."""
        task = self.GSMTask(encode_format='instruct')
        completion = "The answer is 1,000\n#### 1,000"
        
        result = task.extract_gt_answer(completion)
        
        assert result == "1000"  # Comma should be removed
    
    def test_extract_gt_answer_negative(self):
        """Test extraction of negative answer."""
        task = self.GSMTask(encode_format='instruct')
        completion = "The answer is -42\n#### -42"
        
        result = task.extract_gt_answer(completion)
        
        assert result == "-42"
    
    def test_extract_gt_answer_invalid(self):
        """Test extraction with no answer marker."""
        task = self.GSMTask(encode_format='instruct')
        completion = "The answer is 42"  # No #### marker
        
        result = task.extract_gt_answer(completion)
        
        assert result == "[invalid]"
    
    def test_extract_model_answer_instruct(self):
        """Test model answer extraction with instruct format."""
        task = self.GSMTask(encode_format='instruct')
        completion = "The calculation gives us 42 as the final answer."
        
        result, span = task.extract_model_answer(completion)
        
        assert result == "42"
        assert span is not None
        assert isinstance(span, tuple)
        assert len(span) == 2
    
    def test_extract_model_answer_qa(self):
        """Test model answer extraction with qa format."""
        task = self.GSMTask(encode_format='qa')
        completion = "The answer is 42.\nQ: Next question"
        
        result, span = task.extract_model_answer(completion)
        
        assert result == "42"
    
    def test_extract_model_answer_no_number(self):
        """Test model answer extraction with no numbers."""
        task = self.GSMTask(encode_format='instruct')
        completion = "I don't know the answer."
        
        result, span = task.extract_model_answer(completion)
        
        assert result == "[invalid]"
        assert span is None
    
    def test_extract_model_answer_last_number(self):
        """Test that last number is extracted."""
        task = self.GSMTask(encode_format='instruct')
        completion = "First we have 10, then 20, and finally 42."
        
        result, span = task.extract_model_answer(completion)
        
        assert result == "42"
    
    def test_is_correct_true(self):
        """Test is_correct with matching answers."""
        task = self.GSMTask(encode_format='instruct')
        gt_example = {"answer": "The answer is #### 42"}
        model_answer = "42"
        
        result = task.is_correct(gt_example, model_answer)
        
        assert result is True
    
    def test_is_correct_false(self):
        """Test is_correct with non-matching answers."""
        task = self.GSMTask(encode_format='instruct')
        gt_example = {"answer": "The answer is #### 42"}
        model_answer = "43"
        
        result = task.is_correct(gt_example, model_answer)
        
        assert result is False
    
    def test_is_correct_with_comma_normalization(self):
        """Test is_correct handles comma normalization."""
        task = self.GSMTask(encode_format='instruct')
        gt_example = {"answer": "The answer is #### 1,000"}
        model_answer = "1000"
        
        result = task.is_correct(gt_example, model_answer)
        
        assert result is True


class TestDecodingArguments:
    """Test suite for DecodingArguments dataclass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from cot_dataset.cot_decoding.solve import DecodingArguments
        self.DecodingArguments = DecodingArguments
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        args = self.DecodingArguments()
        
        assert args.encode_format == "instruct"
        assert args.max_new_tokens == 512
        assert args.decoding == "greedy"
        assert args.cot_n_branches == 10
        assert args.cot_aggregate == "sum"
    
    def test_custom_values(self):
        """Test setting custom values."""
        args = self.DecodingArguments(
            encode_format="qa",
            max_new_tokens=256,
            decoding="cot",
            cot_n_branches=5,
            cot_aggregate="max"
        )
        
        assert args.encode_format == "qa"
        assert args.max_new_tokens == 256
        assert args.decoding == "cot"
        assert args.cot_n_branches == 5
        assert args.cot_aggregate == "max"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
