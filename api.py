from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
from flask_sqlalchemy import SQLAlchemy
from AprioriDP.AprioriDP import apriori
from The_PAM_Clustering.PAM import PAM
import pandas as pd
import urllib
import pyodbc
import os
import sys

login = sys.argv[1]
pwd = sys.argv[2]

# Flask aplication
app = Flask(__name__)

connection_string = "Driver={ODBC Driver 17 for SQL Server};"\
                    "Server=tcp:opendataserver2020.database.windows.net;"\
                    "Database=OpenDataDbms;"\
                    "uid=stud20;pwd=!Student2020"

cs1 = urllib.parse.quote_plus(connection_string)
pa = "mssql+pyodbc:///?odbc_connect=%s" % cs1

# SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = pa
db = SQLAlchemy(app)

# Flask_restful
api = Api(app)


class DBExperiment(db.Model):
    __tablename__ = 'tw_ayupov_experiments'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    params = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return "<Experiment %d with parameters %s>" % (self.id, self.params)

    def tojson(self):
        return {
            "id": self.id,
            "params": self.params,
        }


class DBClusterResult(db.Model):
    __tablename__ = 'tw_ayupov_clusters'
    id = db.Column(db.Integer, primary_key=True)

    exp_id = db.Column(db.Integer,
                       db.ForeignKey('tw_ayupov_experiments.id'),
                       nullable=False)
    experiments = db.relationship('DBExperiment',
                                  backref=db.backref('clusters', lazy=True))

    stud_id = db.Column(db.Integer)  # TODO change to ForeignKey
    # stud_id = db.Column(db.Integer, db.ForeignKey('stud.id'), nullable=False)
    # stud = db.relationship('Student')  # TODO db from other resource

    cluster = db.Column(db.Integer, nullable=False)
    type = db.Column(db.Boolean, nullable=False)

    def __repr__(self):
        return '''<Cluster experiment %d shows
                student %d is in cluster %d>''' % (self.exp_id,
                                                   self.stud_id,
                                                   self.cluster)

    def tojson(self):
        return {
            "exp_id": self.exp_id,
            "stud_id": self.stud_id,
            "cluster": self.cluster,
            "type": self.type,
        }


class DBRuleResult(db.Model):
    __tablename__ = 'tw_ayupov_rules'
    id = db.Column(db.Integer, primary_key=True)
    exp_id = db.Column(db.Integer,
                       db.ForeignKey('tw_ayupov_experiments.id'),
                       nullable=False)
    experiments = db.relationship('DBExperiment',
                                  backref=db.backref('rules', lazy=True))

    rule = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return '''<Rule experiment %d shows %s>''' % (self.exp_id,
                                                      self.rule)

    def tojson(self):
        return {
            "exp_id": self.exp_id,
            "rule": self.rule,
        }


# Experiment
# shows a single experiment item, can be started once by POST and deleted
class Experiment(Resource):
    def get(self, exp_id):
        exp_in_db = DBExperiment.query.get(exp_id)
        if exp_in_db is None:
            abort(404, message="experiment {} doesn't exist".format(exp_id))

        if exp_in_db.params is None:
            return exp_in_db.tojson()
        elif 'PAM' in exp_in_db.params:
            exps = []
            for exp in DBClusterResult.query.filter_by(exp_id=exp_id):
                exps.append(exp.tojson())
            return jsonify(exps)
        elif 'AprioriDP' in exp_in_db.params:
            for exp in DBRuleResult.query.filter_by(exp_id=exp_id):
                exps.append(exp.tojson())
            return jsonify(exps)
        else:
            raise ValueError

    def delete(self, exp_id):
        exp_in_db = DBExperiment.query.get(exp_id)
        if exp_in_db is None:
            abort(404, message="experiment {} doesn't exist".format(exp_id))
        db.session.delete(exp_in_db)
        db.session.commit()
        return 'Successfully deleted', 204

    def post(self, exp_id):
        parser = reqparse.RequestParser()
        parser.add_argument('algo', required=True, help="algo cannot be blank")
        parser.add_argument('dataset', type=int)
        parser.add_argument('k', type=int)
        parser.add_argument('min_supp', type=float)
        parser.add_argument('min_conf', type=float)
        parser.add_argument('max_iter', type=int)
        args = parser.parse_args()
        experiment = DBExperiment.query.get(exp_id)

        if args['algo'] not in ['PAM', 'AprioriDP']:
            abort(400,
                  message="algo should be one of following: PAM or AprioriDP")

        if experiment is None:
            abort(404, message="no page for this experiment, post one")

        if experiment.params is not None:
            abort(409, message="this experiment is over, start new")

        if args['dataset'] is None:
            abort(400, message="dataset is required")

        if args['algo'] == 'PAM' and args['k'] is None:
            abort(400, message="parameter k is required in PAM")

        if args['algo'] == 'AprioriDP' and (args['min_supp'] is None
                                            or args['min_conf'] is None):
            abort(400, message="min_supp and min_conf are required in apriori")

        output = []
        if args['algo'] == 'PAM':
            param_string = "algo == %s, k == %d" % ('PAM', args['k'])
            dsnum = args['dataset']
            if dsnum == 1:
                dspath = os.path.abspath('./data/tutors_small.csv')
            else:
                dspath = os.path.abspath('./data/tutors.csv')

            ds = pd.read_csv(dspath, sep=';', encoding='utf-8')
            if args['max_iter'] is None:
                param_string += ", maxIter == 10000"
                medidx, clusters, totalDistance = PAM(ds,
                                                      args['k'])
            else:
                param_string += (", maxIter == " + str(args['max_iter']))
                medidx, clusters, totalDistance = PAM(ds,
                                                      args['k'],
                                                      maxIter=args['max_iter'])

            for (item, _cluster) in enumerate(clusters):
                cluster_res = DBClusterResult(experiments=experiment,
                                              cluster=_cluster,
                                              stud_id=item,
                                              type=False)
                if item in medidx:
                    cluster_res.type = True

                db.session.add(cluster_res)
                output.append(cluster_res.tojson())
        else:
            param_format = """algo == %s, min_supp == %f, min_conf == %f"""
            param_tuple = ('AprioriDP', args['min_supp'], args['min_conf'])
            param_string = (param_format % param_tuple)
            freq_subsets, conf_rules = apriori(args['dataset'],
                                               args['min_supp'],
                                               args['min_conf'])
            for _rule in conf_rules:
                rule_str = "(%s => %s, confidence==%f)" % _rule
                rule_res = DBRuleResult(experiments=experiment,
                                        rule=rule_str)
                db.session.add(rule_res)
                output.append(rule_res.tojson())

        experiment.params = param_string
        db.session.commit()
        return jsonify(output), 201


# ExperimentList
# shows a list of all experiments, and lets you POST to add new tasks
class ExperimentList(Resource):
    def get(self):
        exps = []
        for exp in DBExperiment.query.order_by(DBExperiment.id).all():
            exps.append(exp.tojson())
        return jsonify(exps)

    def post(self):
        experiment = DBExperiment(params=None)
        db.session.add(experiment)
        db.session.commit()

        return "Created experiment: " + str(experiment.id), 201


# setup
api.add_resource(ExperimentList, '/experiments',
                 '/', '/experiments/',
                 endpoint='experiments')
api.add_resource(Experiment, '/experiments/<int:exp_id>')


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
