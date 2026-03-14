# Newproject Skill

Create new voiceover matcher projects with proper structure and configuration.

## When to Use

Use this skill when:
- User asks to create a new project
- User mentions "newproject" or "new project"
- User wants to set up a project for voiceover matching

## Workflow

1. **Get Project Details** - Ask for project name, voiceover file, output directory
2. **Create Directory Structure** - Create project folder with required subdirectories
3. **Copy Voiceover** - Place voiceover file in `voiceover/` folder
4. **Create Config** - Generate `project_config.yaml` with defaults
5. **Verify** - Confirm project is ready for pipeline

## Directory Structure

```
project_name/
├── voiceover/
│   └── voiceover.srt (or .mp3/.wav)
├── project_config.yaml
└── (other files created by pipeline)
```

## Usage

```bash
# Interactive mode
python main.py --newproject "E:\Projects\MyDoc"

# Or use CLI
python src/cli/newproject.py "E:\Projects\MyDoc" --voiceover script.srt
```

## Notes

- Uses `config.yaml` defaults from project root
- Project config merges with global config
- Supports `--voiceover` to specify voiceover file
- Supports `--template` for custom config templates
