#!/usr/bin/env python3
"""
Single-file parallel privacy rewriting script
No manual operation required, automatically completes the entire preference dataset construction
"""

import os
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
import sys

def setup_logging():
    """Setup logging"""
    log_dir = './logs/candidate_dataset_construction'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'parallel_preference_dataset_construction_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_batch_range(start_batch, end_batch, args):
    """Run privacy rewriting for specified batch range"""
    logger = logging.getLogger(__name__)
    
    # Build command
    cmd = [
        'python', 'candidate_dataset_construction.py',
        '--qwen_model_path', args.qwen_model_path,
        '--dataset_path', args.dataset_path,
        '--max_candidates', str(args.max_candidates),
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--start_batch', str(start_batch),
        '--end_batch', str(end_batch)
    ]
    
    logger.info(f"Starting to process batches {start_batch}-{end_batch-1}")
    
    try:
        # Run command
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Batches {start_batch}-{end_batch-1} processed successfully")
            return True, f"Batches {start_batch}-{end_batch-1} succeeded"
        else:
            logger.error(f"✗ Batches {start_batch}-{end_batch-1} processing failed")
            logger.error(f"Error message: {result.stderr}")
            return False, f"Batches {start_batch}-{end_batch-1} failed"
            
    except Exception as e:
        logger.error(f"✗ Batches {start_batch}-{end_batch-1} exception: {e}")
        return False, f"Batches {start_batch}-{end_batch-1} exception"

def get_dataset_size(dataset_path):
    """Get dataset size"""
    try:
        from datasets import load_from_disk
        
        # Load entire dataset directly
        dataset = load_from_disk(dataset_path)
        
        # Get training set
        if isinstance(dataset, dict) and 'train' in dataset:
            train_dataset = dataset['train']
        else:
            # If it's directly a Dataset object, use it
            train_dataset = dataset
            
        return len(train_dataset)
    except Exception as e:
        print(f"Failed to get dataset size: {e}")
        # Try using load_dataset as fallback
        try:
            from datasets import load_dataset
            dataset = load_dataset('arrow', data_files=os.path.join(dataset_path, 'train/data-00000-of-00001.arrow'))
            return len(dataset['train'])
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None

def main():
    """Main function"""
    print("=" * 80)
    print("🚀 Parallel Privacy Rewriting System Starting")
    print("=" * 80)

    parser = argparse.ArgumentParser(description='Privacy rewriting system')
    parser.add_argument('--qwen_model_path', type=str, default='./models/Qwen2.5-1.5b-Instruct',
                        help='Path to the Qwen model')
    parser.add_argument('--dataset_path', type=str, default='./datasets/Pri_DDXPlus_SLJA_dpo',
                        help='Path to the input dataset')
    parser.add_argument('--candidate_dataset_path', type=str, default='./datasets/DPO/candidate/Pri_DDXPlus_SLJA_dpo_candidate',
                        help='Path to save the candidate dataset')
    parser.add_argument('--max_candidates', type=int, default=10,
                        help='Maximum number of candidates to generate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for processing')
    parser.add_argument('--parallel_workers', type=int, default=3,
                        help='Number of parallel worker processes')
    parser.add_argument('--batches_per_worker', type=int, default=300,
                        help='Number of batches each worker process handles')
    parser.add_argument('--timeout', type=str, default=None,
                        help='Timeout limit (None for no timeout)')
    
    args = parser.parse_args()

    # Setup logging (must be before any logger usage)
    logger = setup_logging()

    if os.path.exists(args.candidate_dataset_path):
        logger.info(f"Candidate dataset already exists: {args.candidate_dataset_path}")
        return
    
    logger.info("Starting parallel privacy rewriting processing")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Number of parallel worker processes: {args.parallel_workers}")
    logger.info(f"Number of batches per worker: {args.batches_per_worker}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Batch timeout: No limit")
    
    # Check dataset path
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        print(f"❌ Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    # Get dataset size
    print("📊 Analyzing dataset...")
    total_samples = get_dataset_size(args.dataset_path)
    
    if total_samples is None:
        logger.error("Unable to get dataset size")
        print("❌ Error: Unable to get dataset size")
        return
    
    total_batches = (total_samples + args.batch_size - 1) // args.batch_size
    
    print(f"📈 Dataset Information:")
    print(f"   - Total samples: {total_samples:,}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Total batches: {total_batches:,}")
    print(f"   - Parallel processes: {args.parallel_workers}")
    print(f"   - Batches per process: {args.batches_per_worker}")
    
    logger.info(f"Dataset size: {total_samples} samples")
    logger.info(f"Total batches: {total_batches}")
    
    # Generate batch ranges - starting from batch 1
    start_from_batch = 1
    batch_ranges = []
    for i in range(start_from_batch - 1, total_batches, args.batches_per_worker):
        start_batch = i + 1
        end_batch = min(i + args.batches_per_worker + 1, total_batches + 1)
        batch_ranges.append((start_batch, end_batch))
    
    print(f"📋 Batch Allocation (starting from batch {start_from_batch}):")
    for i, (start, end) in enumerate(batch_ranges):
        print(f"   Process {i+1}: batches {start}-{end-1}")
    
    logger.info(f"Starting processing from batch {start_from_batch}")
    logger.info(f"Generated batch ranges: {batch_ranges}")
    
    # Parallel processing
    print("\n🔄 Starting parallel processing...")
    start_time = time.time()
    successful_ranges = []
    failed_ranges = []
    
    with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
        # Submit all tasks
        future_to_range = {
            executor.submit(run_batch_range, start, end, args): (start, end)
            for start, end in batch_ranges
        }
        
        # Process completed tasks
        completed = 0
        total_tasks = len(future_to_range)
        
        for future in as_completed(future_to_range):
            start, end = future_to_range[future]
            completed += 1
            
            try:
                success, message = future.result()
                if success:
                    successful_ranges.append((start, end))
                    print(f"✅ [{completed}/{total_tasks}] {message}")
                else:
                    failed_ranges.append((start, end))
                    print(f"❌ [{completed}/{total_tasks}] {message}")
            except Exception as e:
                failed_ranges.append((start, end))
                print(f"💥 [{completed}/{total_tasks}] Batches {start}-{end-1} exception: {e}")
    
    # Statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("📊 Processing Completion Statistics")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"✅ Successful ranges: {len(successful_ranges)}")
    print(f"❌ Failed ranges: {len(failed_ranges)}")
    print(f"📈 Success rate: {len(successful_ranges)/len(batch_ranges)*100:.1f}%")
    
    if successful_ranges:
        print(f"\n✅ Successful batch ranges:")
        for start, end in successful_ranges:
            print(f"   - Batches {start}-{end-1}")
    
    if failed_ranges:
        print(f"\n❌ Failed batch ranges:")
        for start, end in failed_ranges:
            print(f"   - Batches {start}-{end-1}")
    
    # Calculate processed data volume
    successful_samples = sum((end - start) * args.batch_size for start, end in successful_ranges)
    print(f"\n📊 Data Statistics:")
    print(f"   - Successfully processed samples: {successful_samples:,}")
    print(f"   - Total samples: {total_samples:,}")
    print(f"   - Processing completion rate: {successful_samples/total_samples*100:.1f}%")
    
    logger.info(f"Parallel processing completed, total time: {total_time:.2f} seconds")
    logger.info(f"Successful ranges: {len(successful_ranges)}")
    logger.info(f"Failed ranges: {len(failed_ranges)}")
    
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted processing")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Program exception: {e}")
        sys.exit(1)
