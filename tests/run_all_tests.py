#!/usr/bin/env python3
"""
Comprehensive test runner for RAG system validation.
Runs all test categories with proper reporting and validation.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class RAGTestRunner:
    """Comprehensive test runner for RAG system."""
    
    def __init__(self):
        self.test_categories = {
            'unit': {
                'description': 'Unit tests for individual components',
                'files': [
                    'test_document_processing.py',
                    'test_file_storage.py', 
                    'test_search_engine.py',
                    'test_rag_pipeline.py',
                    'test_rag_config.py'
                ],
                'timeout': 300,  # 5 minutes
                'parallel': True
            },
            'integration': {
                'description': 'End-to-end integration tests',
                'files': [
                    'test_rag_integration.py'
                ],
                'timeout': 600,  # 10 minutes
                'parallel': False
            },
            'performance': {
                'description': 'Performance and memory tests',
                'files': [
                    'test_rag_performance.py'
                ],
                'timeout': 900,  # 15 minutes
                'parallel': False
            },
            'fips': {
                'description': 'FIPS compliance validation',
                'files': [
                    'test_fips_compliance.py'
                ],
                'timeout': 300,  # 5 minutes
                'parallel': True
            },
            'demo': {
                'description': 'Demo scenario validation',
                'files': [
                    'test_demo_scenario.py'
                ],
                'timeout': 600,  # 10 minutes
                'parallel': False
            },
            'app': {
                'description': 'Application integration tests',
                'files': [
                    'test_app_integration.py',
                    'test_session_manager.py',
                    'test_image_processing.py',
                    'test_ollama_client.py',
                    'test_config.py'
                ],
                'timeout': 300,  # 5 minutes
                'parallel': True
            }
        }
        
        self.results = {}
        self.total_start_time = None
    
    def setup_environment(self):
        """Set up test environment."""
        print("Setting up test environment...")
        
        # Ensure test dependencies are available
        try:
            import pytest
            import psutil
            import reportlab
            print("✓ Core test dependencies available")
        except ImportError as e:
            print(f"✗ Missing test dependency: {e}")
            print("Run: pip install -r requirements.txt")
            return False
        
        # Create test fixtures if needed
        fixtures_dir = Path(__file__).parent / "fixtures"
        if not fixtures_dir.exists():
            fixtures_dir.mkdir(exist_ok=True)
            print("✓ Created fixtures directory")
        
        # Set test environment variables
        os.environ["RAG_TEST_MODE"] = "1"
        os.environ["RAG_DISABLE_EXTERNAL_DEPS"] = "1"
        
        return True
    
    def run_test_category(self, category: str, verbose: bool = False) -> Dict:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        config = self.test_categories[category]
        print(f"\n{'='*60}")
        print(f"Running {category.upper()} tests: {config['description']}")
        print(f"{'='*60}")
        
        results = {
            'category': category,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'duration': 0,
            'files': config['files']
        }
        
        start_time = time.time()
        
        for test_file in config['files']:
            print(f"\nRunning {test_file}...")
            
            # Build pytest command
            cmd = [
                sys.executable, "-m", "pytest",
                str(Path(__file__).parent / test_file),
                "-v" if verbose else "-q",
                f"--timeout={config['timeout']}",
                "--tb=short",
                "--disable-warnings"
            ]
            
            # Add parallel execution if supported
            if config['parallel'] and len(config['files']) > 1:
                cmd.extend(["-n", "auto"])
            
            # Add coverage if requested
            if os.environ.get("RAG_TEST_COVERAGE"):
                cmd.extend([
                    "--cov=rag",
                    "--cov-report=term-missing",
                    "--cov-append"
                ])
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config['timeout']
                )
                
                if result.returncode == 0:
                    print(f"✓ {test_file} passed")
                    results['passed'] += 1
                else:
                    print(f"✗ {test_file} failed")
                    results['failed'] += 1
                    results['errors'].append({
                        'file': test_file,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    })
                
                if verbose:
                    print(f"Output:\n{result.stdout}")
                    if result.stderr:
                        print(f"Errors:\n{result.stderr}")
                        
            except subprocess.TimeoutExpired:
                print(f"✗ {test_file} timed out after {config['timeout']}s")
                results['failed'] += 1
                results['errors'].append({
                    'file': test_file,
                    'error': f"Timeout after {config['timeout']}s"
                })
            
            except Exception as e:
                print(f"✗ {test_file} error: {e}")
                results['failed'] += 1
                results['errors'].append({
                    'file': test_file,
                    'error': str(e)
                })
        
        results['duration'] = time.time() - start_time
        return results
    
    def run_all_tests(self, categories: Optional[List[str]] = None, verbose: bool = False) -> Dict:
        """Run all test categories or specified ones."""
        if not self.setup_environment():
            return {'success': False, 'error': 'Environment setup failed'}
        
        self.total_start_time = time.time()
        categories_to_run = categories if categories is not None else list(self.test_categories.keys())
        
        print(f"Starting comprehensive RAG test suite...")
        print(f"Categories to run: {', '.join(categories_to_run)}")
        
        all_results = {
            'success': True,
            'categories': {},
            'summary': {
                'total_passed': 0,
                'total_failed': 0,
                'total_skipped': 0,
                'total_duration': 0,
                'categories_run': len(categories_to_run),
                'categories_passed': 0
            }
        }
        
        for category in categories_to_run:
            try:
                result = self.run_test_category(category, verbose)
                all_results['categories'][category] = result
                
                # Update summary
                all_results['summary']['total_passed'] += result['passed']
                all_results['summary']['total_failed'] += result['failed']
                all_results['summary']['total_skipped'] += result['skipped']
                
                if result['failed'] == 0:
                    all_results['summary']['categories_passed'] += 1
                    print(f"✓ {category.upper()} category passed")
                else:
                    all_results['success'] = False
                    print(f"✗ {category.upper()} category failed")
                    
            except Exception as e:
                print(f"✗ Failed to run {category} tests: {e}")
                all_results['success'] = False
                all_results['categories'][category] = {
                    'category': category,
                    'passed': 0,
                    'failed': 1,
                    'errors': [{'error': str(e)}],
                    'duration': 0
                }
        
        all_results['summary']['total_duration'] = time.time() - self.total_start_time
        return all_results
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append(f"Overall Status: {'PASSED' if results['success'] else 'FAILED'}")
        report.append(f"Total Duration: {summary['total_duration']:.1f} seconds")
        report.append(f"Categories Run: {summary['categories_run']}")
        report.append(f"Categories Passed: {summary['categories_passed']}")
        report.append("")
        
        # Test counts
        report.append("Test Summary:")
        report.append(f"  Passed:  {summary['total_passed']}")
        report.append(f"  Failed:  {summary['total_failed']}")
        report.append(f"  Skipped: {summary['total_skipped']}")
        report.append("")
        
        # Category details
        for category, result in results['categories'].items():
            report.append(f"{category.upper()} Tests:")
            report.append(f"  Status: {'PASSED' if result['failed'] == 0 else 'FAILED'}")
            report.append(f"  Duration: {result['duration']:.1f}s")
            report.append(f"  Files: {', '.join(result['files'])}")
            report.append(f"  Results: {result['passed']} passed, {result['failed']} failed")
            
            if result['errors']:
                report.append("  Errors:")
                for error in result['errors']:
                    report.append(f"    - {error.get('file', 'Unknown')}: {error.get('error', 'Unknown error')}")
            
            report.append("")
        
        # Success criteria validation
        report.append("Success Criteria Validation:")
        criteria = [
            ("Unit test coverage", summary['total_passed'] > 0),
            ("Integration tests pass", results['categories'].get('integration', {}).get('failed', 1) == 0),
            ("Performance tests pass", results['categories'].get('performance', {}).get('failed', 1) == 0),
            ("FIPS compliance verified", results['categories'].get('fips', {}).get('failed', 1) == 0),
            ("Demo scenario validated", results['categories'].get('demo', {}).get('failed', 1) == 0),
        ]
        
        for criterion, passed in criteria:
            status = "✓ PASS" if passed else "✗ FAIL"
            report.append(f"  {criterion}: {status}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, results: Dict, output_file: Optional[str] = None):
        """Save test report to file."""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"test_report_{timestamp}.txt"
        
        report = self.generate_report(results)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"\nTest report saved to: {output_file}")
        return output_file


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="RAG System Test Runner")
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=list(RAGTestRunner().test_categories.keys()),
        help="Test categories to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable code coverage reporting"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for test report"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test subset (unit tests only)"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    if args.coverage:
        os.environ["RAG_TEST_COVERAGE"] = "1"
    
    # Determine categories to run
    if args.quick:
        categories = ["unit"]
    elif args.categories:
        categories = args.categories
    else:
        categories = None  # Run all
    
    # Run tests
    runner = RAGTestRunner()
    results = runner.run_all_tests(categories, args.verbose)
    
    # Generate and display report
    report = runner.generate_report(results)
    print("\n" + report)
    
    # Save report if requested
    if args.output or not results['success']:
        runner.save_report(results, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main() 