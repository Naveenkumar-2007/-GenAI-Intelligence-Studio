"""Product-specific tools for ReAct agent in Product Builder mode."""

from typing import List
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel


def build_feature_generator_tool(llm: BaseLanguageModel) -> Tool:
    """Generate MVP features for a product idea."""
    
    def _generate_features(product_idea: str) -> str:
        prompt = f"""You are a product manager. For the following product idea, generate:
1. 5 MUST-HAVE MVP features (critical for launch)
2. 5 NICE-TO-HAVE features (can wait for v2)
3. 3 features to AVOID in MVP (scope creep)

Product Idea: {product_idea}

Format as bullet points:"""
        
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    
    return Tool(
        name="feature_generator",
        description="Generate MVP feature list for a product idea. Input: product description.",
        func=_generate_features,
    )


def build_user_persona_tool(llm: BaseLanguageModel) -> Tool:
    """Generate user personas for a product."""
    
    def _generate_personas(product_idea: str) -> str:
        prompt = f"""You are a UX researcher. For the following product idea, create 3 detailed user personas:

Product Idea: {product_idea}

For each persona include:
- Name & Demographics
- Goals & Motivations  
- Pain Points
- Tech Savviness (1-5)
- Key Quote
- How they'd use this product

Format clearly with headers:"""
        
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    
    return Tool(
        name="user_persona_generator",
        description="Generate detailed user personas for a product. Input: product description.",
        func=_generate_personas,
    )


def build_system_architect_tool(llm: BaseLanguageModel) -> Tool:
    """Design system architecture for a product."""
    
    def _design_architecture(requirements: str) -> str:
        prompt = f"""You are a senior system architect. Design a scalable architecture for:

Requirements: {requirements}

Include:
1. **Frontend**: Framework, key components
2. **Backend**: Language, framework, API design
3. **Database**: Type (SQL/NoSQL), main tables/collections
4. **Infrastructure**: Cloud services, deployment strategy
5. **Security**: Authentication, data protection
6. **Scalability**: How to handle growth

Provide a clear, practical architecture:"""
        
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    
    return Tool(
        name="system_architect",
        description="Design system architecture for a product. Input: product requirements or features.",
        func=_design_architecture,
    )


def build_competitor_analyzer_tool(llm: BaseLanguageModel) -> Tool:
    """Analyze potential competitors and market positioning."""
    
    def _analyze_competitors(product_idea: str) -> str:
        prompt = f"""You are a market analyst. For the following product idea:

Product Idea: {product_idea}

Provide:
1. **Likely Competitors**: 3-5 existing products in this space
2. **Competitive Advantages**: What could make this product stand out
3. **Market Gaps**: Opportunities competitors are missing
4. **Pricing Strategy**: Suggested pricing model
5. **Go-to-Market**: Initial launch strategy

Be specific and actionable:"""
        
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    
    return Tool(
        name="competitor_analyzer",
        description="Analyze competitors and market positioning for a product idea.",
        func=_analyze_competitors,
    )


def build_tech_stack_recommender_tool(llm: BaseLanguageModel) -> Tool:
    """Recommend technology stack for a product."""
    
    def _recommend_stack(product_requirements: str) -> str:
        prompt = f"""You are a tech lead. Recommend the best technology stack for:

Requirements: {product_requirements}

Provide:
1. **Frontend**: Framework + libraries (with reasoning)
2. **Backend**: Language + framework (with reasoning)
3. **Database**: Primary + cache (with reasoning)
4. **Hosting**: Cloud provider + services
5. **DevOps**: CI/CD, monitoring, logging
6. **Estimated Cost**: Monthly cost for MVP scale

Consider: developer availability, learning curve, scalability, cost:"""
        
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    
    return Tool(
        name="tech_stack_recommender",
        description="Recommend technology stack for a product. Input: product requirements.",
        func=_recommend_stack,
    )


def build_product_tools(llm: BaseLanguageModel) -> List[Tool]:
    """Build all product-specific tools for ReAct agent."""
    return [
        build_feature_generator_tool(llm),
        build_user_persona_tool(llm),
        build_system_architect_tool(llm),
        build_competitor_analyzer_tool(llm),
        build_tech_stack_recommender_tool(llm),
    ]
