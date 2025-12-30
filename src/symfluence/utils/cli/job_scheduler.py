import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

class JobScheduler:
    """
    Manages job submission to SLURM and other schedulers.
    """
    
    def __init__(self):
        """Initialize the JobScheduler."""
        pass

    def handle_slurm_job_submission(self, execution_plan: Dict[str, Any],
                                     symfluence_instance: Optional[Any] = None) -> bool:
        """
        Handle SLURM job submission workflow.
        
        Args:
            execution_plan: Execution plan from CLI manager
            symfluence_instance: SYMFLUENCE instance (optional)
            
        Returns:
            bool: Success status
        """
        try:
            slurm_options = execution_plan.get('slurm_options', {})
            
            if not slurm_options.get('job_account'):
                print("⚠️ Warning: No SLURM account specified. This may be required on your system.")
                print("   Use --job_account to specify an account if job submission fails.")
            
            result = self.submit_slurm_job(execution_plan, symfluence_instance)
            
            if result.get('success', False):
                print(f"\n🎉 SLURM job submission successful!")
                if not slurm_options.get('submit_and_wait', False):
                    print(f"💡 Job is running in background. Monitor with:")
                    print(f"   squeue -j {result['job_id']}")
                    print(f"   tail -f SYMFLUENCE_*_{result['job_id']}.out")
                return True
            else:
                print(f"❌ Job submission failed")
                return False
        
        except Exception as e:
            print(f"❌ Error in SLURM job submission: {str(e)}")
            return False

    def submit_slurm_job(self, execution_plan: Dict[str, Any], symfluence_instance=None) -> Dict[str, Any]:
        """
        Submit a SLURM job for the execution plan.
        """
        print("\n🚀 Preparing SLURM Job Submission:")
        print("=" * 60)
        
        if not self._check_slurm_available():
            raise RuntimeError("SLURM commands (sbatch) not available on this system")
        
        slurm_options = execution_plan.get('slurm_options', {})
        
        # Auto-generate job name if not provided
        if not slurm_options.get('job_name'):
            if symfluence_instance and hasattr(symfluence_instance, 'config'):
                domain = symfluence_instance.config.get('DOMAIN_NAME', 'symfluence')
            else:
                domain = 'symfluence'
            
            mode = execution_plan.get('mode', 'workflow')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            slurm_options['job_name'] = f"{domain}_{mode}_{timestamp}"
        
        print(f"🏷️  Job Name: {slurm_options['job_name']}")
        print(f"⏱️  Time Limit: {slurm_options['job_time']}")
        print(f"💾 Memory: {slurm_options['job_memory']}")
        print(f"🔢 Tasks: {slurm_options['job_ntasks']}")
        
        if slurm_options.get('job_account'):
            print(f"💳 Account: {slurm_options['job_account']}")
        if slurm_options.get('job_partition'):
            print(f"📊 Partition: {slurm_options['job_partition']}")
        
        config_file = execution_plan.get('config_file', './0_config_files/config_template.yaml')
        
        script_content = self._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )
        
        script_path = Path(f"SYMFLUENCE_{slurm_options['job_name']}.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 SLURM script created: {script_path}")
        
        try:
            print(f"\n🚀 Submitting job...")
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            job_id = result.stdout.strip().split()[-1]
            
            if not job_id:
                raise RuntimeError("Could not extract job ID from sbatch output")
            
            print(f"✅ Job submitted successfully!")
            print(f"🆔 Job ID: {job_id}")
            
            submission_result = {
                'success': True,
                'job_id': job_id,
                'script_path': script_path,
                'job_name': slurm_options['job_name'],
                'submission_time': datetime.now().isoformat()
            }
            
            if slurm_options.get('submit_and_wait', False):
                print(f"\n⏳ Waiting for job {job_id} to complete...")
                self._monitor_slurm_job(job_id)
            
            return submission_result
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Error submitting job: {e}")
            print(f"stderr: {e.stderr}")
            raise RuntimeError(f"Failed to submit SLURM job: {e}")

    def _create_symfluence_slurm_script(self, execution_plan: Dict[str, Any],
                                        slurm_options: Dict[str, Any],
                                        config_file: str) -> str:
        """Create SLURM script content for SYMFLUENCE workflow."""
        job_mode = execution_plan.get('job_mode', 'workflow')
        job_steps = execution_plan.get('job_steps', [])
        
        if job_mode == 'individual_steps':
            symfluence_cmd = f"python SYMFLUENCE.py --config {config_file}"
            for step in job_steps:
                symfluence_cmd += f" --{step}"
        elif job_mode == 'pour_point_setup':
            pour_point_info = execution_plan.get('pour_point', {})
            symfluence_cmd = (
                f"python SYMFLUENCE.py "
                f"--pour_point {pour_point_info.get('coordinates')} "
                f"--domain_def {pour_point_info.get('domain_definition_method')} "
                f"--domain_name '{pour_point_info.get('domain_name')}'"
            )
            if pour_point_info.get('bounding_box_coords'):
                symfluence_cmd += f" --bounding_box_coords {pour_point_info['bounding_box_coords']}"
        else:
            symfluence_cmd = f"python SYMFLUENCE.py --config {config_file}"
        
        settings = execution_plan.get('settings', {})
        if settings.get('force_rerun', False):
            symfluence_cmd += " --force_rerun"
        if settings.get('debug', False):
            symfluence_cmd += " --debug"
        if settings.get('continue_on_error', False):
            symfluence_cmd += " --continue_on_error"
        
        script = f"""#!/bin/bash
#SBATCH --job-name={slurm_options['job_name']}
#SBATCH --output=SYMFLUENCE_{slurm_options['job_name']}_%j.out
#SBATCH --error=SYMFLUENCE_{slurm_options['job_name']}_%j.err
#SBATCH --time={slurm_options['job_time']}
#SBATCH --ntasks={slurm_options['job_ntasks']}
#SBATCH --mem={slurm_options['job_memory']} """
        
        if slurm_options.get('job_account'):
            script += f"\n#SBATCH --account={slurm_options['job_account']}"
        if slurm_options.get('job_partition'):
            script += f"\n#SBATCH --partition={slurm_options['job_partition']}"
        if slurm_options.get('job_nodes') and slurm_options['job_nodes'] > 1:
            script += f"\n#SBATCH --nodes={slurm_options['job_nodes']}"
        
        script += f"""

# Job information
echo "=========================================="
echo "SYMFLUENCE SLURM Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {slurm_options['job_name']}"
echo "Node: $HOSTNAME"
echo "Started: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules and environment
echo "Loading environment..."
"""
        
        if slurm_options.get('job_modules'):
            script += f"module restore {slurm_options['job_modules']}\n"
        
        if slurm_options.get('conda_env'):
            script += f"conda activate {slurm_options['conda_env']}\n"
        
        script += f"""
echo "Python environment: $(which python)"
echo "SYMFLUENCE directory: $(pwd)"
echo ""

# Run SYMFLUENCE workflow
echo "Starting SYMFLUENCE workflow..."
echo "Command: {symfluence_cmd}"
echo ""

{symfluence_cmd}

exit_code=$?

echo ""
echo "=========================================="
echo "SYMFLUENCE Job Completed"
echo "Exit code: $exit_code"
echo "Finished: $(date)"
echo "=========================================="

exit $exit_code
"""
        return script
    
    def _check_slurm_available(self) -> bool:
        """Check if SLURM commands are available."""
        return shutil.which('sbatch') is not None
    
    def _monitor_slurm_job(self, job_id: str) -> None:
        """Monitor SLURM job until completion."""
        import time
        print(f"🔄 Monitoring job {job_id}...")
        while True:
            try:
                result = subprocess.run(
                    ['squeue', '-j', job_id, '-h'],
                    capture_output=True, text=True
                )
                if not result.stdout.strip():
                    print(f"✅ Job {job_id} completed!")
                    break
                else:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        status_info = lines[0].split()
                        if len(status_info) >= 5:
                            status = status_info[4]
                            print(f"⏳ Job status: {status}")
            except subprocess.SubprocessError as e:
                print(f"⚠️ Error checking job status: {e}")
            time.sleep(60)

    def detect_environment(self) -> str:
        """
        Detect whether we're running on HPC or a personal computer.
        """
        hpc_schedulers = ['sbatch', 'qsub', 'bsub']
        for scheduler in hpc_schedulers:
            if shutil.which(scheduler):
                return 'hpc'
        
        hpc_env_vars = [
            'SLURM_CLUSTER_NAME', 'SLURM_JOB_ID', 'PBS_JOBID',
            'SGE_CLUSTER_NAME', 'LOADL_STEP_ID'
        ]
        
        for env_var in hpc_env_vars:
            if env_var in os.environ:
                return 'hpc'
        
        if Path('/scratch').exists():
            return 'hpc'
        
        return 'laptop'
