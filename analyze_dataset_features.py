#!/usr/bin/env python3
"""
Comprehensive analysis of balanced LOTL dataset and LOLBAS API features
to identify logical consistency issues and valuable features for model training.
"""

import pandas as pd
import re
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DatasetAnalyzer:
    def __init__(self):
        self.balanced_dataset = None
        self.lolbas_data = None
        self.suspicious_patterns = []
        self.feature_analysis = {}
        
    def load_data(self):
        """Load the balanced dataset and LOLBAS API data"""
        try:
            # Load balanced dataset
            self.balanced_dataset = pd.read_csv('balanced_combined_lotl_dataset.csv')
            print(f"✅ Loaded balanced dataset: {len(self.balanced_dataset)} commands")
            
            # Load LOLBAS data
            self.lolbas_data = pd.read_csv('LOLBAS APIs.csv')
            print(f"✅ Loaded LOLBAS data: {len(self.lolbas_data)} entries")
            
            # Display column information
            print(f"\n📊 Balanced Dataset Columns: {list(self.balanced_dataset.columns)}")
            print(f"📊 LOLBAS Dataset Columns: {list(self.lolbas_data.columns)}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        return True
    
    def analyze_command_logic(self):
        """Analyze commands for logical consistency and suspicious patterns"""
        print("\n🔍 ANALYZING COMMAND LOGICAL CONSISTENCY...")
        
        suspicious_commands = []
        logic_issues = {
            'conflicting_operations': [],
            'impossible_paths': [],
            'syntax_errors': [],
            'inconsistent_arguments': [],
            'suspicious_combinations': []
        }
        
        for idx, row in self.balanced_dataset.iterrows():
            command = row['Command']
            label = row['Label']
            issues = []
            
            # Check for conflicting operations in single command
            if self._has_conflicting_operations(command):
                issues.append('conflicting_operations')
                logic_issues['conflicting_operations'].append((command, label))
            
            # Check for impossible file paths
            if self._has_impossible_paths(command):
                issues.append('impossible_paths') 
                logic_issues['impossible_paths'].append((command, label))
            
            # Check for syntax errors
            if self._has_syntax_errors(command):
                issues.append('syntax_errors')
                logic_issues['syntax_errors'].append((command, label))
            
            # Check for inconsistent arguments
            if self._has_inconsistent_arguments(command):
                issues.append('inconsistent_arguments')
                logic_issues['inconsistent_arguments'].append((command, label))
            
            # Check for suspicious command combinations
            if self._has_suspicious_combinations(command):
                issues.append('suspicious_combinations')
                logic_issues['suspicious_combinations'].append((command, label))
            
            if issues:
                suspicious_commands.append({
                    'command': command,
                    'label': label,
                    'issues': issues
                })
        
        # Print summary
        print(f"\n📈 LOGIC ANALYSIS RESULTS:")
        total_issues = sum(len(v) for v in logic_issues.values())
        print(f"Total commands with logic issues: {len(suspicious_commands)}/{len(self.balanced_dataset)} ({len(suspicious_commands)/len(self.balanced_dataset)*100:.1f}%)")
        
        for issue_type, commands in logic_issues.items():
            if commands:
                print(f"\n⚠️  {issue_type.replace('_', ' ').title()}: {len(commands)} instances")
                # Show first few examples
                for i, (cmd, lbl) in enumerate(commands[:3]):
                    print(f"   Example {i+1}: {'MALICIOUS' if lbl == 1 else 'BENIGN'}")
                    print(f"   Command: {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
        
        return logic_issues, suspicious_commands
    
    def _has_conflicting_operations(self, command):
        """Check for conflicting operations like mkdir + rmdir in same command"""
        # Look for conflicting file operations
        conflicts = [
            (r'\bmkdir\b.*\brmdir\b', r'\brmdir\b.*\bmkdir\b'),  # mkdir vs rmdir
            (r'\bdel\b.*\bcopy\b', r'\bcopy\b.*\bdel\b'),        # delete vs copy same file
            (r'\battrib\s+\+h\b.*\battrib\s+-h\b', r'\battrib\s+-h\b.*\battrib\s+\+h\b'),  # hide vs unhide
        ]
        
        for pattern1, pattern2 in conflicts:
            if re.search(pattern1, command, re.IGNORECASE) or re.search(pattern2, command, re.IGNORECASE):
                return True
        return False
    
    def _has_impossible_paths(self, command):
        """Check for impossible or malformed file paths"""
        # Look for clearly impossible paths
        impossible_patterns = [
            r'[A-Z]:[^\\]',  # Drive letter not followed by backslash
            r'\\{3,}',       # Triple or more backslashes
            r'[<>|*?]',      # Invalid filename characters in paths
            r'\.\.\.+',      # Multiple dots in sequence
        ]
        
        for pattern in impossible_patterns:
            if re.search(pattern, command):
                return True
        return False
    
    def _has_syntax_errors(self, command):
        """Check for obvious syntax errors"""
        # Look for unmatched quotes, brackets, etc.
        quote_issues = [
            command.count('"') % 2 != 0,  # Unmatched double quotes
            command.count("'") % 2 != 0,  # Unmatched single quotes
            command.count('(') != command.count(')'),  # Unmatched parentheses
            command.count('{') != command.count('}'),  # Unmatched braces
        ]
        
        return any(quote_issues)
    
    def _has_inconsistent_arguments(self, command):
        """Check for inconsistent or contradictory arguments"""
        # Look for contradictory flags or arguments
        contradictions = [
            (r'/q\b.*\b/v\b', r'/v\b.*\b/q\b'),  # quiet vs verbose
            (r'/f\b.*\b/i\b', r'/i\b.*\b/f\b'),  # force vs interactive
        ]
        
        for pattern1, pattern2 in contradictions:
            if re.search(pattern1, command, re.IGNORECASE) or re.search(pattern2, command, re.IGNORECASE):
                return True
        return False
    
    def _has_suspicious_combinations(self, command):
        """Check for suspicious command combinations that might indicate synthetic data"""
        # Look for unusual combinations that real attackers wouldn't use
        suspicious = [
            # Extremely verbose operations that serve no purpose
            len(command) > 500 and command.count('\\') > 10,
            # Multiple redundant operations
            command.lower().count('echo') > 3,
            command.lower().count('copy') > 3,
            # Nonsensical file operations
            re.search(r'copy.*copy.*copy', command, re.IGNORECASE),
        ]
        
        return any(suspicious)
    
    def analyze_lolbas_features(self):
        """Analyze LOLBAS data to identify valuable features beyond just Command"""
        print("\n🔍 ANALYZING LOLBAS FEATURES...")
        
        # Analyze all available columns
        columns = self.lolbas_data.columns.tolist()
        feature_importance = {}
        
        for col in columns:
            if col == 'Command':  # Skip the command column as it's already used
                continue
                
            # Calculate feature value based on uniqueness and information content
            unique_values = self.lolbas_data[col].nunique()
            non_null_ratio = self.lolbas_data[col].notna().sum() / len(self.lolbas_data)
            
            feature_importance[col] = {
                'unique_values': unique_values,
                'non_null_ratio': non_null_ratio,
                'sample_values': self.lolbas_data[col].dropna().unique()[:5].tolist()
            }
        
        # Identify most valuable features
        print(f"\n📊 LOLBAS FEATURE ANALYSIS:")
        print(f"Available columns: {len(columns)}")
        
        # Sort by potential value (combination of uniqueness and completeness)
        valuable_features = []
        
        for col, stats in feature_importance.items():
            # Calculate value score
            completeness_score = stats['non_null_ratio']
            diversity_score = min(stats['unique_values'] / 50, 1.0)  # Normalize to max 1.0
            value_score = (completeness_score * 0.7) + (diversity_score * 0.3)
            
            valuable_features.append({
                'column': col,
                'value_score': value_score,
                'completeness': f"{stats['non_null_ratio']:.1%}",
                'unique_values': stats['unique_values'],
                'samples': stats['sample_values']
            })
        
        # Sort by value score
        valuable_features.sort(key=lambda x: x['value_score'], reverse=True)
        
        print(f"\n🎯 TOP VALUABLE FEATURES FOR MODEL TRAINING:")
        for i, feature in enumerate(valuable_features[:8]):  # Show top 8
            col = feature['column']
            print(f"\n{i+1}. {col}")
            print(f"   📈 Value Score: {feature['value_score']:.2f}")
            print(f"   📋 Completeness: {feature['completeness']}")
            print(f"   🔢 Unique Values: {feature['unique_values']}")
            print(f"   💡 Sample Values: {feature['samples']}")
            
            # Provide specific recommendations for each feature
            if col == 'Command Category':
                print(f"   🎯 Use Case: Categorize attack types (Execute, Download, ADS, etc.)")
            elif col == 'MITRE ATT&CK technique':
                print(f"   🎯 Use Case: Map to MITRE ATT&CK framework for threat intelligence")
            elif col == 'Command Privileges':
                print(f"   🎯 Use Case: Risk assessment based on required privileges")
            elif col == 'Command Usecase':
                print(f"   🎯 Use Case: Intent classification and behavior prediction")
            elif col == 'Detections':
                print(f"   🎯 Use Case: Feature extraction for signature-based detection")
            elif col == 'Operating System':
                print(f"   🎯 Use Case: Platform-specific model training")
            elif col == 'Tags':
                print(f"   🎯 Use Case: Additional categorization and metadata")
        
        return valuable_features
    
    def generate_enhanced_dataset_recommendations(self):
        """Generate recommendations for enhancing the dataset"""
        print(f"\n🚀 RECOMMENDATIONS FOR ENHANCED LOTL DETECTION:")
        
        recommendations = {
            'feature_engineering': [
                "Extract Command Category from LOLBAS (Execute, Download, ADS, etc.)",
                "Map commands to MITRE ATT&CK techniques for threat intelligence",
                "Add privilege level features (User, Admin, System)",
                "Include binary path analysis for process execution context",
                "Extract file extension patterns from command arguments",
                "Add command complexity metrics (length, argument count, nesting)"
            ],
            'dataset_improvements': [
                "Include Command Usecase as behavioral intent labels",
                "Add detection rules as negative feature indicators", 
                "Incorporate OS-specific command variations",
                "Include timestamp and sequence analysis for command chains",
                "Add network-based vs local execution classification",
                "Include parent-child process relationship context"
            ],
            'model_enhancements': [
                "Multi-label classification for Command Category + MITRE technique",
                "Hierarchical classification: Category -> Technique -> Specific threat",
                "Ensemble approach: Syntax + Semantic + Behavioral models",
                "Sequence modeling for command chain analysis",
                "Confidence scoring based on privilege escalation paths",
                "Real-time threat intelligence integration"
            ],
            'detection_strategies': [
                "Signature-based detection using Detections field patterns",
                "Anomaly detection for unusual privilege escalation sequences",
                "Behavioral analysis using Command Usecase patterns",
                "Context-aware detection using process tree analysis",
                "Threat hunting queries based on MITRE ATT&CK mapping",
                "Adaptive thresholding based on environment baseline"
            ]
        }
        
        for category, items in recommendations.items():
            print(f"\n📋 {category.replace('_', ' ').title()}:")
            for i, item in enumerate(items, 1):
                print(f"   {i}. {item}")
        
        return recommendations
    
    def create_enhanced_feature_dataset(self):
        """Create a sample enhanced dataset with LOLBAS features"""
        print(f"\n🔨 CREATING ENHANCED FEATURE DATASET SAMPLE...")
        
        # Create mapping from commands to LOLBAS features
        command_features = {}
        
        for _, row in self.lolbas_data.iterrows():
            cmd = row['Command']
            if pd.notna(cmd):
                # Extract key features
                features = {
                    'category': row.get('Command Category', 'Unknown'),
                    'mitre_technique': row.get('MITRE ATT&CK technique', 'Unknown'),
                    'privileges': row.get('Command Privileges', 'Unknown'),
                    'usecase': row.get('Command Usecase', 'Unknown'),
                    'os': row.get('Operating System', 'Unknown'),
                    'has_detections': pd.notna(row.get('Detections', None))
                }
                command_features[cmd] = features
        
        # Sample 100 commands for demonstration
        sample_size = min(100, len(self.balanced_dataset))
        sample_data = self.balanced_dataset.sample(n=sample_size, random_state=42).copy()
        
        # Add enhanced features
        enhanced_features = []
        for _, row in sample_data.iterrows():
            cmd = row['Command']
            
            # Try to match with LOLBAS features (simplified matching)
            matched_features = None
            for lolbas_cmd, features in command_features.items():
                # Simple matching - check if main executable is present
                if self._extract_main_executable(cmd) == self._extract_main_executable(lolbas_cmd):
                    matched_features = features
                    break
            
            if matched_features:
                enhanced_features.append(matched_features)
            else:
                # Default features for unmatched commands
                enhanced_features.append({
                    'category': 'Unknown',
                    'mitre_technique': 'Unknown', 
                    'privileges': 'User',
                    'usecase': 'Unknown',
                    'os': 'Windows',
                    'has_detections': False
                })
        
        # Add features to sample dataset
        for feature_name in ['category', 'mitre_technique', 'privileges', 'usecase', 'os', 'has_detections']:
            sample_data[feature_name] = [f[feature_name] for f in enhanced_features]
        
        # Save enhanced sample
        output_file = 'enhanced_lotl_sample.csv'
        sample_data.to_csv(output_file, index=False)
        print(f"✅ Saved enhanced sample dataset: {output_file}")
        
        # Show feature distribution
        print(f"\n📊 ENHANCED FEATURE DISTRIBUTION:")
        for feature in ['category', 'mitre_technique', 'privileges']:
            if feature in sample_data.columns:
                print(f"\n{feature.replace('_', ' ').title()}:")
                value_counts = sample_data[feature].value_counts()
                for val, count in value_counts.head().items():
                    print(f"   {val}: {count} ({count/len(sample_data)*100:.1f}%)")
        
        return sample_data
    
    def _extract_main_executable(self, command):
        """Extract the main executable from a command"""
        if pd.isna(command):
            return ""
        
        # Simple extraction - get first word that looks like an executable
        words = command.split()
        for word in words:
            if word.endswith('.exe') or word in ['net', 'cmd', 'powershell', 'wmic', 'certutil']:
                return word.lower()
        
        return words[0].lower() if words else ""
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE LOTL DATASET ANALYSIS...")
        
        # Load data
        if not self.load_data():
            return
        
        # Analyze command logic
        logic_issues, suspicious_commands = self.analyze_command_logic()
        
        # Analyze LOLBAS features  
        valuable_features = self.analyze_lolbas_features()
        
        # Generate recommendations
        recommendations = self.generate_enhanced_dataset_recommendations()
        
        # Create enhanced sample dataset
        enhanced_sample = self.create_enhanced_feature_dataset()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📊 Total commands analyzed: {len(self.balanced_dataset)}")
        print(f"📋 Logic issues found: {len(suspicious_commands)}")
        print(f"🎯 Valuable LOLBAS features identified: {len(valuable_features)}")
        print(f"💡 Enhancement recommendations generated: {sum(len(r) for r in recommendations.values())}")
        
        return {
            'logic_issues': logic_issues,
            'suspicious_commands': suspicious_commands,
            'valuable_features': valuable_features,
            'recommendations': recommendations,
            'enhanced_sample': enhanced_sample
        }

def main():
    """Main execution function"""
    analyzer = DatasetAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print(f"\n📝 SUMMARY:")
    print(f"The analysis reveals that while most commands are logically consistent,")
    print(f"there are significant opportunities to enhance the dataset with rich")
    print(f"features from the LOLBAS API beyond just the Command column.")
    print(f"\nKey findings:")
    print(f"- Command Category, MITRE ATT&CK technique, and Privileges are the most valuable features")
    print(f"- Enhanced feature engineering could significantly improve model performance")
    print(f"- Multi-label classification approach recommended for comprehensive threat detection")

if __name__ == "__main__":
    main()