import unittest
import os

import utils.arguments_parser

class TestArgumentsParser(unittest.TestCase):
    def test_parse_cli_args(self):
        arguments = ['-t', 'hvalues', '-e', 'blackjack', '-n', '1000', '--noprint']
        parsed_args = utils.arguments_parser.parse_args(arguments)
        self.assertEqual(parsed_args['technique'], 'hvalues')
        self.assertEqual(parsed_args['environment'], 'blackjack')
        self.assertEqual(parsed_args['num_episodes'], 1000)
        self.assertEqual(parsed_args['should_print'], False)

    def test_parse_file_args(self):
        """Test parsing arguments from a "complete" JSON input file."""
        arguments = ['-f', 'tests/files/full.json']
        parsed_args = utils.arguments_parser.parse_args(arguments)
        self.assertEqual(parsed_args['technique'], 'hvalues')
        self.assertEqual(parsed_args['environment'], 'blackjack')
        self.assertEqual(parsed_args['num_episodes'], 100000)
        self.assertEqual(parsed_args['should_print'], True)

    def test_parse_file_args_missing_agent(self):
        """Test parsing arguments from a JSON input file that is missing the agent."""
        arguments = ['-f', 'tests/files/missing_agent.json']
        parsed_args = utils.arguments_parser.parse_args(arguments)
        self.assertEqual(parsed_args['technique'], 'hvalues')
        self.assertEqual(parsed_args['environment'], 'blackjack')
        self.assertEqual(parsed_args['num_episodes'], 100000)
        self.assertEqual(parsed_args['should_print'], True)
        self.assertEqual(parsed_args['agent'], {})

    def test_parse_file_args_missing_env(self):
        """Test parsing arguments from a JSON input file that is missing the environment params (only has env params)."""
        arguments = ['-f', 'tests/files/only_agent.json']
        parsed_args = utils.arguments_parser.parse_args(arguments)
        self.assertEqual(parsed_args['technique'], 'hvalues')
        self.assertEqual(parsed_args['environment'], 'blackjack')
        self.assertEqual(parsed_args['num_episodes'], 100000)
        self.assertEqual(parsed_args['should_print'], True)

    def test_save_params(self):
        """Test saving parameters to a JSON file."""
        arguments = ['-t', 'hvalues', '-e', 'blackjack', '-n', '1000', '--noprint']
        parsed_args = utils.arguments_parser.parse_args(arguments)
        path = utils.arguments_parser.save_params(parsed_args)
        assert os.path.exists(path)
        os.remove(path)
