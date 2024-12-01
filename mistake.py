import subprocess

# Command to run (replace with your desired command)
command = "echo 'Hello from Zsh!'"

# Run the command using the default shell (which is usually Bash or Zsh)
result = subprocess.run(command, shell=True, text=True, capture_output=True)

# Print the output (stdout)
print(result.stdout)

# Check if there was any error (stderr)
if result.stderr:
    print(f"Error: {result.stderr}")
