from .build_op import Stage as BuildOp

def get_build_stage(name, **kwargs):
    return {
        'build_op': BuildOp
    }[name](**kwargs)