#!/bin/bash

# Initialize the Git repository
git init

# Create or reset the file to start with
echo "Initial commit" > file.txt
git add file.txt
git commit -m "Initial commit"
echo "Creating random commit density..."

# Set the number of days and maximum commits per day
num_days=365
max_commits_per_day=5

# Loop through each day of the year
for day in $(seq 0 $((num_days - 1))); do
    # Get a random number of commits for today, limited by max_commits_per_day
    num_commits_today=$((RANDOM % (max_commits_per_day + 1)))

    # Loop for the number of commits to make today
    for ((i = 0; i < num_commits_today; i++)); do
        # Append a line to the file and commit it
        echo "Commit for day $day, commit $i" >> file.txt
        git add file.txt

        # Generate a random date for the commit
        # Randomly select hour, minute, and second to ensure unique timestamps
        hour=$(printf "%02d" $((RANDOM % 24)))
        minute=$(printf "%02d" $((RANDOM % 60)))
        second=$(printf "%02d" $((RANDOM % 60)))

        # Set the commit date using the current day and random time
        commit_date=$(date -d "$day days ago $hour:$minute:$second" '+%Y-%m-%d %H:%M:%S')

        # Commit with the generated random date
        git commit -m "Random commit for day $day" --date="$commit_date"
    done
done

echo "Random commit density has been created."

# Optionally, you can add your GitHub remote and push the commits
# git remote add origin https://github.com/YourUsername/my-random-density-repo.git
# git push -u origin master
