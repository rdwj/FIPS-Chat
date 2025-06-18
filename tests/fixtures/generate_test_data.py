"""
Test data generation utilities.
Creates sample PDFs and mock data for comprehensive RAG testing.
"""

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER

# Sample content templates for different document types
DOCUMENT_TEMPLATES = {
    "machine_learning": {
        "title": "Machine Learning Fundamentals",
        "sections": [
            {
                "title": "Introduction to Machine Learning",
                "content": """
                Machine learning is a subset of artificial intelligence (AI) that provides systems 
                the ability to automatically learn and improve from experience without being 
                explicitly programmed. Machine learning focuses on the development of computer 
                programs that can access data and use it to learn for themselves.
                
                The process of learning begins with observations or data, such as examples, 
                direct experience, or instruction, in order to look for patterns in data and 
                make better decisions in the future based on the examples that we provide.
                """
            },
            {
                "title": "Types of Machine Learning",
                "content": """
                There are several types of machine learning algorithms:
                
                1. Supervised Learning: Uses labeled training data to learn a mapping function 
                from input variables to output variables. Examples include classification and 
                regression problems.
                
                2. Unsupervised Learning: Uses unlabeled data to discover hidden patterns. 
                Examples include clustering and dimensionality reduction.
                
                3. Reinforcement Learning: An agent learns to make decisions by taking actions 
                in an environment to maximize cumulative reward.
                """
            },
            {
                "title": "Neural Networks",
                "content": """
                Neural networks are computing systems inspired by biological neural networks. 
                They consist of interconnected nodes (neurons) that process information using 
                a connectionist approach to computation.
                
                Deep learning is a subset of machine learning that uses neural networks with 
                multiple layers (deep neural networks) to model and understand complex patterns 
                in data. These networks can learn hierarchical representations of data.
                """
            }
        ]
    },
    "data_science": {
        "title": "Data Science and Analytics",
        "sections": [
            {
                "title": "Introduction to Data Science",
                "content": """
                Data science is an interdisciplinary field that uses scientific methods, 
                processes, algorithms and systems to extract knowledge and insights from 
                structured and unstructured data. It employs techniques and theories drawn 
                from many fields including mathematics, statistics, computer science, and 
                information science.
                """
            },
            {
                "title": "Data Processing Pipeline",
                "content": """
                The data science process typically follows these steps:
                
                1. Data Collection: Gathering data from various sources including databases, 
                APIs, web scraping, and sensors.
                
                2. Data Cleaning: Identifying and correcting errors, handling missing values, 
                and removing duplicates.
                
                3. Data Exploration: Understanding the data through visualization and 
                statistical analysis.
                
                4. Feature Engineering: Creating new features from existing data to improve 
                model performance.
                
                5. Model Development: Building and training machine learning models.
                
                6. Model Evaluation: Testing model performance using appropriate metrics.
                
                7. Deployment: Putting the model into production for real-world use.
                """
            }
        ]
    },
    "security": {
        "title": "Information Security and FIPS Compliance",
        "sections": [
            {
                "title": "FIPS Standards Overview",
                "content": """
                Federal Information Processing Standards (FIPS) are publicly announced 
                standards developed by the United States federal government for use in 
                computer systems by non-military American government agencies and government 
                contractors.
                
                FIPS 140-2 is a U.S. government computer security standard used to approve 
                cryptographic modules. The standard provides four increasing, qualitative 
                levels of security intended to cover a wide range of potential applications 
                and environments.
                """
            },
            {
                "title": "Cryptographic Hash Functions",
                "content": """
                A cryptographic hash function is a mathematical algorithm that maps data of 
                arbitrary size to a bit array of a fixed size. The values returned by a 
                hash function are called hash values, hash codes, digests, or simply hashes.
                
                FIPS-approved hash functions include:
                - SHA-256: Produces a 256-bit hash value
                - SHA-384: Produces a 384-bit hash value  
                - SHA-512: Produces a 512-bit hash value
                
                MD5 is not FIPS-approved and should not be used in FIPS-compliant systems.
                """
            },
            {
                "title": "Data Privacy Regulations",
                "content": """
                Modern data privacy regulations such as GDPR (General Data Protection 
                Regulation) and CCPA (California Consumer Privacy Act) require organizations 
                to implement appropriate technical and organizational measures to protect 
                personal data.
                
                Key principles include:
                - Data minimization: Collect only necessary data
                - Purpose limitation: Use data only for stated purposes
                - Storage limitation: Keep data only as long as necessary
                - Security: Implement appropriate security measures
                """
            }
        ]
    }
}

class TestDataGenerator:
    """Generates test data for RAG system validation."""
    
    def __init__(self, fixtures_dir: str = "tests/fixtures"):
        self.fixtures_dir = Path(fixtures_dir)
        self.fixtures_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Create custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12
        )
        
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
    
    def generate_sample_pdf(self, template_key: str, filename: str, pages_target: int = 4) -> str:
        """Generate a sample PDF document."""
        template = DOCUMENT_TEMPLATES[template_key]
        pdf_path = self.fixtures_dir / filename
        
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Add title
        story.append(Paragraph(template["title"], self.title_style))
        story.append(Spacer(1, 12))
        
        # Add sections
        for section in template["sections"]:
            story.append(Paragraph(section["title"], self.heading_style))
            
            # Split content into paragraphs and add some filler to reach target pages
            paragraphs = section["content"].strip().split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), self.body_style))
                    story.append(Spacer(1, 6))
            
            # Add some filler content to reach target page count
            if len(story) < pages_target * 15:  # Rough estimate of elements per page
                filler_content = f"""
                This is additional content for the {section['title'].lower()} section. 
                This content is designed to extend the document to meet the required page count 
                for testing purposes. It includes technical details and explanations that would 
                be found in a comprehensive guide on this topic.
                
                The implementation considerations include scalability, performance optimization, 
                and security best practices. Modern systems must balance functionality with 
                security requirements, especially in regulated environments that require 
                compliance with standards such as FIPS 140-2.
                """
                story.append(Paragraph(filler_content, self.body_style))
                story.append(Spacer(1, 12))
        
        doc.build(story)
        return str(pdf_path)
    
    def generate_demo_test_set(self) -> List[str]:
        """Generate the full demo test set (75 PDFs, 300 pages total)."""
        pdf_files = []
        
        # Calculate pages per document to reach 300 total pages
        base_pages = 300 // 75  # 4 pages per doc
        extra_pages = 300 % 75   # Additional pages to distribute
        
        template_keys = list(DOCUMENT_TEMPLATES.keys())
        
        for i in range(75):
            # Distribute extra pages among first documents
            pages_for_doc = base_pages + (1 if i < extra_pages else 0)
            
            # Cycle through templates
            template_key = template_keys[i % len(template_keys)]
            
            # Generate filename
            filename = f"demo_doc_{i+1:02d}_{template_key}.pdf"
            
            # Generate the PDF
            pdf_path = self.generate_sample_pdf(template_key, filename, pages_for_doc)
            pdf_files.append(pdf_path)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/75 demo documents...")
        
        return pdf_files
    
    def generate_test_queries(self) -> List[Dict[str, Any]]:
        """Generate test queries for different scenarios."""
        return [
            {
                "query": "What is machine learning?",
                "expected_topics": ["machine learning", "artificial intelligence", "algorithms"],
                "difficulty": "basic"
            },
            {
                "query": "Explain the difference between supervised and unsupervised learning",
                "expected_topics": ["supervised learning", "unsupervised learning", "labeled data"],
                "difficulty": "intermediate"
            },
            {
                "query": "How do neural networks process information?",
                "expected_topics": ["neural networks", "neurons", "deep learning"],
                "difficulty": "intermediate"
            },
            {
                "query": "What are FIPS compliance requirements for cryptographic hash functions?",
                "expected_topics": ["FIPS", "cryptographic", "hash functions", "SHA-256"],
                "difficulty": "advanced"
            },
            {
                "query": "Describe the data science pipeline from collection to deployment",
                "expected_topics": ["data science", "pipeline", "data collection", "model deployment"],
                "difficulty": "advanced"
            },
            {
                "query": "What security measures are required for GDPR compliance?",
                "expected_topics": ["GDPR", "data privacy", "security measures", "personal data"],
                "difficulty": "advanced"
            },
            {
                "query": "How does reinforcement learning differ from other ML approaches?",
                "expected_topics": ["reinforcement learning", "agent", "reward", "environment"],
                "difficulty": "intermediate"
            },
            {
                "query": "What is feature engineering in data science?",
                "expected_topics": ["feature engineering", "data processing", "model performance"],
                "difficulty": "intermediate"
            }
        ]
    
    def create_mock_rag_responses(self) -> List[Dict[str, Any]]:
        """Create expected RAG response templates for validation."""
        return [
            {
                "query": "What is machine learning?",
                "expected_response_contains": [
                    "subset of artificial intelligence",
                    "learn from experience",
                    "without being explicitly programmed"
                ],
                "expected_sources": ["machine_learning"],
                "min_relevance_score": 0.7
            },
            {
                "query": "What are FIPS standards?",
                "expected_response_contains": [
                    "Federal Information Processing Standards",
                    "government computer security standard",
                    "cryptographic modules"
                ],
                "expected_sources": ["security"],
                "min_relevance_score": 0.6
            }
        ]
    
    def save_test_metadata(self, pdf_files: List[str], queries: List[Dict], responses: List[Dict]):
        """Save test metadata for validation."""
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_documents": len(pdf_files),
            "total_pages": 300,  # Target for demo
            "document_files": [Path(f).name for f in pdf_files],
            "test_queries": queries,
            "expected_responses": responses,
            "templates_used": list(DOCUMENT_TEMPLATES.keys())
        }
        
        metadata_path = self.fixtures_dir / "test_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_path)


def main():
    """Generate all test data for RAG validation."""
    print("Generating RAG test data...")
    
    generator = TestDataGenerator()
    
    # Generate demo document set
    print("Creating demo document set (75 PDFs, 300 pages)...")
    pdf_files = generator.generate_demo_test_set()
    
    # Generate test queries
    queries = generator.generate_test_queries()
    
    # Generate expected responses
    responses = generator.create_mock_rag_responses()
    
    # Save metadata
    metadata_path = generator.save_test_metadata(pdf_files, queries, responses)
    
    print(f"\nTest data generation complete!")
    print(f"Generated {len(pdf_files)} PDF files")
    print(f"Generated {len(queries)} test queries")
    print(f"Metadata saved to: {metadata_path}")
    print(f"All files saved to: {generator.fixtures_dir}")


if __name__ == "__main__":
    main() 