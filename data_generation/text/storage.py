"""
Memory bank storage system for managing generated utterances.
Provides JSON-based storage with metadata and retrieval capabilities.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

class MemoryBankStorage:
    """Handles storage and retrieval of generated utterances."""
    
    def __init__(self, data_dir: str = "memory_bank/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create data subdirectories
        for domain in ["medical", "legal", "financial", "technical"]:
            (self.data_dir / domain).mkdir(exist_ok=True)
    
    def save_utterances(self, domain: str, utterances: List[Dict[str, Any]], 
                       batch_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save utterances to domain-specific storage.
        
        Args:
            domain: Domain name
            utterances: List of utterance dictionaries
            batch_metadata: Optional metadata about this batch
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_utterances_{timestamp}.json"
        filepath = self.data_dir / domain / filename
        
        # Prepare data structure
        data = {
            "metadata": {
                "domain": domain,
                "batch_size": len(utterances),
                "generation_date": datetime.utcnow().isoformat() + 'Z',
                "filename": filename,
                "model_version": "claude-3.5-sonnet",
                **(batch_metadata or {})
            },
            "utterances": utterances
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(utterances)} utterances to {filepath}")
        return str(filepath)
    
    def load_utterances(self, domain: str, filename: str = None) -> Dict[str, Any]:
        """
        Load utterances from storage.
        
        Args:
            domain: Domain name
            filename: Specific filename. If None, loads most recent.
            
        Returns:
            Dictionary with metadata and utterances
        """
        domain_dir = self.data_dir / domain
        
        if filename:
            filepath = domain_dir / filename
        else:
            # Find most recent file
            files = list(domain_dir.glob(f"{domain}_utterances_*.json"))
            if not files:
                raise FileNotFoundError(f"No utterance files found for domain: {domain}")
            filepath = max(files, key=os.path.getmtime)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_all_utterances(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all utterances for a domain across all files.
        
        Args:
            domain: Domain name
            
        Returns:
            List of all utterances for the domain
        """
        domain_dir = self.data_dir / domain
        all_utterances = []
        
        files = list(domain_dir.glob(f"{domain}_utterances_*.json"))
        for filepath in sorted(files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_utterances.extend(data.get('utterances', []))
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue
        
        return all_utterances
    
    def create_master_dataset(self, output_filename: str = None) -> str:
        """
        Create a master dataset combining all domains.
        
        Args:
            output_filename: Output filename. If None, auto-generates.
            
        Returns:
            Path to master dataset file
        """
        if output_filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_filename = f"agentic_asr_dataset_{timestamp}.json"
        
        output_path = self.data_dir / output_filename
        
        # Collect all utterances by domain
        master_data = {
            "metadata": {
                "dataset_name": "Agentic ASR Synthetic Utterances",
                "creation_date": datetime.utcnow().isoformat() + 'Z',
                "model_version": "claude-3.5-sonnet",
                "domains": ["medical", "legal", "financial", "technical"],
                "total_utterances": 0,
                "utterances_per_domain": {}
            }
        }
        
        # Load utterances for each domain
        for domain in ["medical", "legal", "financial", "technical"]:
            try:
                utterances = self.get_all_utterances(domain)
                master_data[domain] = utterances
                master_data["metadata"]["utterances_per_domain"][domain] = len(utterances)
                master_data["metadata"]["total_utterances"] += len(utterances)
                print(f"Loaded {len(utterances)} utterances for {domain}")
            except FileNotFoundError:
                print(f"No utterances found for {domain} domain")
                master_data[domain] = []
                master_data["metadata"]["utterances_per_domain"][domain] = 0
        
        # Save master dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created master dataset: {output_path}")
        print(f"Total utterances: {master_data['metadata']['total_utterances']}")
        
        return str(output_path)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about stored datasets."""
        stats = {
            "domains": {},
            "total_files": 0,
            "total_utterances": 0,
            "storage_size_mb": 0.0
        }
        
        for domain in ["medical", "legal", "financial", "technical"]:
            domain_dir = self.data_dir / domain
            files = list(domain_dir.glob(f"{domain}_utterances_*.json"))
            
            domain_stats = {
                "file_count": len(files),
                "utterance_count": 0,
                "latest_file": None,
                "files": []
            }
            
            for filepath in files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    utterance_count = len(data.get('utterances', []))
                    file_size = filepath.stat().st_size / (1024 * 1024)  # MB
                    
                    domain_stats["utterance_count"] += utterance_count
                    domain_stats["files"].append({
                        "filename": filepath.name,
                        "utterances": utterance_count,
                        "size_mb": round(file_size, 2),
                        "created": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                    })
                    
                    stats["storage_size_mb"] += file_size
                    
                except Exception as e:
                    print(f"Warning: Could not analyze {filepath}: {e}")
            
            # Sort files by creation time
            domain_stats["files"].sort(key=lambda x: x["created"], reverse=True)
            if domain_stats["files"]:
                domain_stats["latest_file"] = domain_stats["files"][0]["filename"]
            
            stats["domains"][domain] = domain_stats
            stats["total_files"] += domain_stats["file_count"]
            stats["total_utterances"] += domain_stats["utterance_count"]
        
        stats["storage_size_mb"] = round(stats["storage_size_mb"], 2)
        return stats
    
    def export_for_research(self, format_type: str = "csv", output_dir: str = None) -> List[str]:
        """
        Export utterances in research-friendly formats.
        
        Args:
            format_type: Export format ("csv", "tsv", "jsonl")
            output_dir: Output directory. If None, uses data_dir.
            
        Returns:
            List of created file paths
        """
        if output_dir is None:
            output_dir = self.data_dir / "exports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        created_files = []
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        for domain in ["medical", "legal", "financial", "technical"]:
            try:
                utterances = self.get_all_utterances(domain)
                if not utterances:
                    continue
                
                if format_type == "csv":
                    import csv
                    filename = f"{domain}_utterances_{timestamp}.csv"
                    filepath = output_dir / filename
                    
                    with open(filepath, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            'id', 'profile', 'utterance', 'raw_form', 'asr_difficulty',
                            'error_targets', 'domain', 'generation_timestamp'
                        ])
                        writer.writeheader()
                        
                        for utterance in utterances:
                            row = utterance.copy()
                            row['error_targets'] = '|'.join(row.get('error_targets', []))
                            writer.writerow(row)
                    
                    created_files.append(str(filepath))
                
                elif format_type == "jsonl":
                    filename = f"{domain}_utterances_{timestamp}.jsonl"
                    filepath = output_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        for utterance in utterances:
                            f.write(json.dumps(utterance, ensure_ascii=False) + '\n')
                    
                    created_files.append(str(filepath))
                
            except FileNotFoundError:
                continue
        
        return created_files
    
    def cleanup_old_files(self, keep_latest: int = 5):
        """
        Clean up old utterance files, keeping only the most recent.
        
        Args:
            keep_latest: Number of latest files to keep per domain
        """
        for domain in ["medical", "legal", "financial", "technical"]:
            domain_dir = self.data_dir / domain
            files = list(domain_dir.glob(f"{domain}_utterances_*.json"))
            
            if len(files) <= keep_latest:
                continue
            
            # Sort by modification time, newest first
            files.sort(key=os.path.getmtime, reverse=True)
            
            # Remove older files
            for filepath in files[keep_latest:]:
                try:
                    filepath.unlink()
                    print(f"Removed old file: {filepath}")
                except Exception as e:
                    print(f"Could not remove {filepath}: {e}")

# Singleton instance
memory_bank_storage = MemoryBankStorage()

# Convenience functions
def save_utterances(domain: str, utterances: List[Dict[str, Any]], 
                   batch_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to save utterances."""
    return memory_bank_storage.save_utterances(domain, utterances, batch_metadata)

def load_utterances(domain: str, filename: str = None) -> Dict[str, Any]:
    """Convenience function to load utterances."""
    return memory_bank_storage.load_utterances(domain, filename)

def create_master_dataset(output_filename: str = None) -> str:
    """Convenience function to create master dataset."""
    return memory_bank_storage.create_master_dataset(output_filename)

def get_dataset_stats() -> Dict[str, Any]:
    """Convenience function to get dataset statistics."""
    return memory_bank_storage.get_dataset_stats()
