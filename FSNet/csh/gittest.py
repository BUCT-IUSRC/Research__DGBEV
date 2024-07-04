import git

# print(cfg.path.base_path)
repo = git.Repo("/home/dell/csh/FSNet-master/")
writer.add_text("git/git_show", styling_git_info(repo))
writer.flush()