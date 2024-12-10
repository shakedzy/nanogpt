import pkg_resources


def path_to_resource_file(name: str) -> str:
    return pkg_resources.resource_filename("nanogpt", f"__resources__/{name}")
