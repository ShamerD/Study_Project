from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_sqlalchemy import SQLAlchemy
from AprioriDP.AprioriDP import apriori
from The_PAM_Clustering.PAM import PAM

# Flask aplication
app = Flask(__name__)

# SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db\\experiments.db'
db = SQLAlchemy(app)

# Flask_restful
api = Api(app)

experiments = {}


class DBExperiment(db.Model):
    __tablename__ = 'experiments'
    id = db.Column(db.Integer, primary_key=True)
    params = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return "<Experiment %d with parameters %s>" % (self.id, self.params)


class DBClusterResult(db.Model):
    __tablename__ = 'cluster_results'
    id = db.Column(db.Integer, primary_key=True)

    exp_id = db.Column(db.Integer,
                       db.ForeignKey('experiments.id'), nullable=False)
    experiments = db.relationship('DBExperiment',
                                  backref=db.backref('clusters', lazy=True))

    # stud_id = db.Column(db.Integer, db.ForeignKey('stud.id'), nullable=False)
    # stud = db.relationship('Student')  # TODO db from other resource

    cluster = db.Column(db.Integer, nullable=False)
    type = db.Column(db.Boolean, nullable=False)

    def __repr__(self):
        return '''<Cluster experiment %d shows
                student %d is in cluster %d>''' % (self.exp_id,
                                                   self.stud_id,
                                                   self.cluster)


class DBRuleResult(db.Model):
    __tablename__ = 'rule_results'
    id = db.Column(db.Integer, primary_key=True)
    exp_id = db.Column(db.Integer,
                       db.ForeignKey('experiments.id'), nullable=False)
    experiments = db.relationship('DBExperiment',
                                  backref=db.backref('rules', lazy=True))

    rule = db.Column(db.Text, nullable=False)


def check_if_exp_exists(exp_id):
    if exp_id not in experiments:
        abort(404, message="experiment {} doesn't exist".format(exp_id))


# Experiment
# shows a single experiment item, can be started once by POST and deleted
class Experiment(Resource):
    def get(self, exp_id):
        check_if_exp_exists(exp_id)
        return experiments[exp_id]

    def delete(self, exp_id):
        check_if_exp_exists(exp_id)
        del experiments[exp_id]
        return '', 204

    def post(self, exp_id):
        parser = reqparse.RequestParser()
        parser.add_argument('algo', required=True, help="algo cannot be blank")
        parser.add_argument('dataset')  # TODO what type
        parser.add_argument('k', type=int)
        parser.add_argument('min_supp', type=float)
        parser.add_argument('min_conf', type=float)
        parser.add_argument('max_iter', type=int)
        args = parser.parse_args()

        if args['algo'] not in ['PAM', 'AprioriDP']:
            abort(400,
                  message="algo should be one of following: PAM or AprioriDP")

        if experiments[exp_id] != {}:
            abort(409, message="this experiment is over, start new")

        if args['dataset'] is None:
            abort(400, message="dataset is required")

        if args['algo'] == 'PAM' and args['k'] is None:
            abort(400, message="parameter k is required in PAM")

        if args['algo'] == 'AprioriDP' and (args['min_supp'] is None
                                            or args['min_conf'] is None):
            abort(400, message="min_supp and min_conf are required")

        output = {}
        if args['algo'] == 'PAM':
            medoids, cluster, totalDistance = PAM(args['dataset'], args['k'])
            # work with output
        else:
            freq_subsets, conf_rules = apriori(args['dataset'],
                                               args['min_supp'],
                                               args['min_conf'])
            # work with output
        experiments[exp_id] = output
        return output, 201


# ExperimentList
# shows a list of all experiments, and lets you POST to add new tasks
class ExperimentList(Resource):
    def get(self):
        return experiments

    def post(self):
        if (len(experiments.keys()) == 0):
            exp_id = 1
        else:
            exp_id = int(max(experiments.keys())) + 1
        experiments[exp_id] = {}
        return experiments[exp_id], 201


# setup
api.add_resource(ExperimentList, '/experiments')
api.add_resource(Experiment, '/experiments/<int:exp_id>')


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
