from local_ocpa.ocpa.algo.discovery.ocpn.versions import inductive
from local_ocpa.ocpa.objects.log.ocel import OCEL
from local_ocpa.ocpa.objects.log.variants.obj import ObjectCentricEventLog
import local_ocpa.ocpa.objects.log.converter.factory as convert_factory

INDUCTIVE = "inductive"

VERSIONS = {INDUCTIVE: inductive.apply}


def apply(ocel, variant=INDUCTIVE, parameters=None):
    if type(ocel) == OCEL:
        return VERSIONS[variant](ocel.log.log, parameters=parameters)
    if type(ocel) == ObjectCentricEventLog:
        df, _ = convert_factory.apply(ocel, variant='json_to_csv')
        return VERSIONS[variant](df, parameters=parameters)
    else:
        return VERSIONS[variant](ocel, parameters=parameters)
