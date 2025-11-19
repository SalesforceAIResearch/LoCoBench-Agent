# LoCoBench-Agent

**An Interactive Benchmark for LLM Agents in Long-Context Software Engineering**

[![Paper](https://img.shields.io/badge/arXiv-2511.13998-b31b1b.svg)](https://arxiv.org/pdf/2511.13998)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

LoCoBench-Agent is a comprehensive evaluation framework for assessing LLM agents in realistic, long-context software engineering workflows. Built upon [LoCoBench](https://github.com/SalesforceAIResearch/LoCoBench), it transforms 8,000 static scenarios into interactive multi-turn agent environments.

## 🌟 Key Features

- **8,000 Interactive Scenarios** across 10 programming languages and 36 domains
- **Multi-Turn Evaluation** supporting up to 50 conversation turns per scenario
- **Long-Context Assessment** spanning 10K-1M tokens with intelligent memory management
- **8 Specialized Tools** including file operations, semantic search, and code analysis
- **9 Bias-Free Metrics** rigorously validated to eliminate file count bias and hierarchy violations
- **Comprehensive Coverage** across 8 task categories (architectural understanding, bug investigation, feature implementation, etc.)

## 📊 Benchmark Statistics

- **Total Scenarios**: 8,000
- **Unique Projects**: 1,000
- **Context Range**: 10K-1M tokens
- **Languages**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, PHP, Ruby, C#
- **Difficulty Levels**: Easy, Medium, Hard, Expert (25% each)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/SalesforceAIResearch/LoCoBench-Agent.git
cd LoCoBench-Agent
pip install -r requirements.txt
```

### Download Evaluation Data

Download the complete evaluation dataset (data.zip):

```bash
# Download data.zip from Google Drive
# Visit: https://drive.google.com/file/d/1HwPztd0bipUUi8zs7Pxo3StZCOnJBwVR/view?usp=sharing
# Or use gdown (install with: pip install gdown)
gdown https://drive.google.com/uc?id=1HwPztd0bipUUi8zs7Pxo3StZCOnJBwVR

# Extract the data
unzip data.zip

# This will create the data/ directory with all evaluation scenarios
```

### Environment Setup

1. **Configure API Keys**

Create an `api.sh` file (gitignored) with your LLM API credentials:

```bash
# Copy the template
cp api.sh.template api.sh

# Edit api.sh with your API keys
export OPENAI_API_KEY="your_openai_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export GOOGLE_API_KEY="your_google_key_here"

# Source the file
source api.sh
```

### Run Evaluation

```bash
# Evaluate a single model
source api.sh && locobench evaluate --mode agent --agent-type openai --model gpt-4.1-mini --scenario-count 30 --context-management adaptive --max-concurrent-scenarios 10 --resume
```


## 🎯 Evaluation Metrics

### Comprehension Metrics (5)
- **Execution Success Rate**: Tool diversity and successful usage patterns
- **Multi-Session Memory Retention**: Context retention across conversation turns
- **Cross-File Consistency**: Naming conventions and import patterns
- **Dependency Traversal**: Import resolution and reference validity
- **Solution Usability**: Code maintainability and readability

### Efficiency Metrics (4)
- **Runtime Efficiency**: Time complexity through algorithmic pattern analysis
- **Memory Efficiency**: Space complexity and memory pattern detection
- **Information Coverage**: Ratio of files accessed to files modified
- **Long-Range Dependency Resolution**: Read-before-write patterns


## 📖 Citation

If you use LoCoBench-Agent in your research, please cite:

```bibtex
@article{Qiu2025LoCoBenchAgentAI,
  title={LoCoBench-Agent: An Interactive Benchmark for LLM Agents in Long-Context Software Engineering},
  author={Qiu, Jielin and Liu, Zuxin and Liu, Zhiwei and Murthy, Rithesh and Zhang, Jianguo and Chen, Haolin and Wang, Shiyu and Zhu, Ming and Yang, Liangwei and Tan, Juntao and Ram, Roshan and Prabhakar, Akshara and Awalgaonkar, Tulika and Chen, Zixiang and Cen, Zhepeng and Qian, Cheng and Heinecke, Shelby and Yao, Weiran and Savarese, Silvio and Xiong, Caiming and Wang, Huan},
  journal={arXiv preprint arXiv:2511.13998},
  year={2025}
}
```

## 🔗 Related Projects

- [LoCoBench](https://github.com/SalesforceAIResearch/LoCoBench) - Long-context code understanding benchmark

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/SalesforceAIResearch/LoCoBench/blob/main/LICENSE.txt) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions or feedback, please open an issue.

---

**Salesforce AI Research** | [Website](https://www.salesforceairesearch.com/)

