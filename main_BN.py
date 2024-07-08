import local_ocpa.ocpa.algo.predictive_monitoring.factory
from gnn_utils import *
from local_ocpa.ocpa.objects.log.importer.csv import factory as csv_import_factory
from local_ocpa.ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from local_ocpa.ocpa.algo.predictive_monitoring import factory as predictive_monitoring
from local_ocpa.ocpa.objects.log.util import misc as log_util
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from graph_embedding import convert_to_nx_graphs, embed
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from tqdm import tqdm

params = {"sap": {"batch_size": 4, "lr": 0.001, "epochs": 15}}

NEXT_ACTIVITY = "next_activity"

def next_activity(node, ocel, params):
    act = params[0]
    e_id = node.event_id
    out_edges = ocel.graph.eog.out_edges(e_id)
    next_act = 0
    for (source, target) in out_edges:
        if ocel.get_value(target, "event_activity") == act:
            next_act = 1
            for (source_, target_) in out_edges:
                if ocel.get_value(target_, "event_timestamp") < ocel.get_value(target, "event_timestamp"):
                    next_act = 0
    return next_act


NEXT_TIMESTAMP = "next_timestamp"


def next_timestamp(node, ocel, params):
    e_id = node.event_id
    out_edges = ocel.graph.eog.out_edges(e_id)
    if len(out_edges) == 0:
        # placeholder, will not be used for prediction
        return 0
    return min([ocel.get_value(target, "event_timestamp") for (source, target) in out_edges]).to_pydatetime().timestamp() - ocel.get_value(e_id, "event_timestamp").to_pydatetime().timestamp()


REL_ACTIVITY_OCCURRENCE_PER_TYPE ="act_occurrence_p_type"
def relative_activity_occurence(node,ocel,params):
    ots = params[0]
    acts = params[1]
    results_dict = {(ot,act):0 for (ot,act) in itertools.product(ots, acts)}
    e_id = node.event_id
    case = ocel.process_executions[node.pexec_id]
    pexec_objects = ocel.process_execution_objects[node.pexec_id]
    oc_counter = {act:{(ot_,o):False for (ot_,o) in pexec_objects} for act in acts}
    this_timestamp = ocel.get_value(e_id,"event_timestamp")
    for e in case:
        if ocel.get_value(e,"event_timestamp") > this_timestamp:
            continue
        else:
            curr_act = ocel.get_value(e,"event_activity")
            ev_ob_dict = {}
            for ot_iter in ots:
                ev_ob_dict[ot_iter] = [o for o in ocel.get_value(e_id,ot_iter)]
            for (ot_,o) in pexec_objects:
                if o in ev_ob_dict[ot_]:
                    oc_counter[curr_act][(ot_,o)] = True
    rel_counter = 0
    for curr_act in acts:
        for curr_ot in ots:
            curr_ot_obs = [(ot_,o) for (ot_,o) in pexec_objects if ot_ == curr_ot]
            for (ot_,o) in curr_ot_obs:
                if oc_counter[curr_act][(ot_,o)]:
                    rel_counter += 1
            results_dict[(curr_ot,curr_act)] = rel_counter/len(curr_ot_obs)

    return results_dict

local_ocpa.ocpa.algo.predictive_monitoring.factory.VERSIONS[
    local_ocpa.ocpa.algo.predictive_monitoring.factory.EVENT_BASED][REL_ACTIVITY_OCCURRENCE_PER_TYPE] = relative_activity_occurence
local_ocpa.ocpa.algo.predictive_monitoring.factory.VERSIONS[
    local_ocpa.ocpa.algo.predictive_monitoring.factory.EVENT_BASED][NEXT_ACTIVITY] = next_activity
local_ocpa.ocpa.algo.predictive_monitoring.factory.VERSIONS[
    local_ocpa.ocpa.algo.predictive_monitoring.factory.EVENT_BASED][NEXT_TIMESTAMP] = next_timestamp

def get_dataframe_from_graph(graph):
    df = pd.read_csv(filename)
    lst_data = {}
    for i in range(len(graph)):
        lst_index = graph[i].ndata['event_indices'].tolist()
        lst_values = graph[i].ndata[predictive_monitoring.EVENT_REMAINING_TIME].tolist()
        lst_index_rt = {k: v for k, v in zip(lst_index, lst_values)}
        lst_data.update(lst_index_rt)
    temp_df = df.loc[df['event_id'].isin(lst_data.keys())]
    temp_df[predictive_monitoring.EVENT_REMAINING_TIME] = temp_df['event_id'].map(lst_data)
    temp_df = temp_df.dropna()
    return temp_df

def Bayesian_nets_prediction(x_train, x_val, x_test):

    x_train_df = get_dataframe_from_graph(x_train)
    # x_val_df = get_dataframe_from_graph(x_val)
    x_test_df = get_dataframe_from_graph(x_test)

    numerical_cols = x_train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = x_train_df.select_dtypes(include=["object"]).columns.tolist()

    # Rergetmove the ta variable from the numerical columns list
    numerical_cols.remove(predictive_monitoring.EVENT_REMAINING_TIME)

    # Define transformers in the pipeline
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("label_encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Final pipeline with adaboost as the estimator
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", AdaBoostRegressor()),
        ]
    )

    X = x_train_df.drop(predictive_monitoring.EVENT_REMAINING_TIME, axis=1)
    Y = x_train_df[predictive_monitoring.EVENT_REMAINING_TIME]

    # Fitting 
    pipeline.fit(X, Y)

    # Predictions
    prediction = pipeline.predict(x_test_df.drop(predictive_monitoring.EVENT_REMAINING_TIME, axis=1))

    # Calculate the score
    score = mean_absolute_error(x_test_df[predictive_monitoring.EVENT_REMAINING_TIME], prediction)

    return score
#
filename = "BPI2017-Final.csv"
lr = 0.01
batch_size = 256
object_types = ["application", "offer"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}
ocel = csv_import_factory.apply(file_path=filename, parameters=parameters)
ks = [2, 3, 4, 5, 6, 7, 8]
# ks = [i for i in range(8,1,-1)]

print("Number of process executions: "+str(len(ocel.process_executions)))
print("Average lengths: " +
      str(sum([len(e) for e in ocel.process_executions])/len(ocel.process_executions)))
activities = list(set(ocel.log.log["event_activity"].tolist()))
print(str(len(activities))+" actvities")
accuracy_dict = {}

for target in [(predictive_monitoring.EVENT_REMAINING_TIME, ())]:

    include_last = True
    if target == (NEXT_TIMESTAMP, ()):
        include_last = False

    F = [target,
         (predictive_monitoring.EVENT_SYNCHRONIZATION_TIME, ())]
    feature_storage = predictive_monitoring.apply(ocel, F, [])

    # replace synchronization time with 1 as placeholder for empty feature
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1
    feature_storage.extract_normalized_train_test_split(0.3, state=3)
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1

    g_set_list_t = []
    g_set_list_te = []
    seq_set_list = []
    seq_set_list_v = []
    seq_set_list_t = []

    for k in ks:
        if True:
            print("___________________________")
            print("Prediction with Bayesian Networks")
            print("___________________________")

            layer_size = len(F)-1

             # generate training & test datasets
            train_idx, val_idx = train_test_split(
                feature_storage.training_indices, test_size=0.2)
            x_train, y_train = generate_graph_dataset(
                feature_storage.feature_graphs, train_idx, ocel, k=k, target=target, include_last=include_last)
            x_val, y_val = generate_graph_dataset(
                feature_storage.feature_graphs, val_idx, ocel, k=k, target=target, include_last=include_last)
            x_test, y_test = generate_graph_dataset(
                feature_storage.feature_graphs, feature_storage.test_indices, ocel, k=k, target=target, include_last=include_last)
            start_time = time.time()
            final_score = Bayesian_nets_prediction(x_train, x_val, x_test)
            calc_time = time.time() - start_time
            # We save the same score for all the metrics to fit the format of the other methods but only the test_MAE is relevant
            accuracy_dict[target[0]+'graph_gnn_k_' + str(k)] = {
                'baseline_MAE': final_score,
                'train_MAE': final_score,
                'val_MAE': final_score,
                'test_MAE': final_score,
                "time":calc_time
            }
            print(pd.DataFrame(accuracy_dict))

#pd.set_option('display.max_columns', None)
print(pd.DataFrame(accuracy_dict))
pd.DataFrame(accuracy_dict).to_csv("results_BN/metrics_BN.csv")
