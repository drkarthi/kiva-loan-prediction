<template>
  <v-container>
    <v-layout
      text-xs-center
      wrap
    >
      <v-flex>
        <!-- IMPORTANT PART! -->
<form>
          <v-text-field
            v-model="loanAmount"
            label="Loan Amount"
            required
          ></v-text-field>
<v-btn @click="submit">submit</v-btn>
          <v-btn @click="clear">clear</v-btn>
        </form>
<br/>
        <br/>
<h1>Predicted Class is: {{ predictedClass }}</h1>
<!-- END: IMPORTANT PART! -->
      </v-flex>
    </v-layout>
  </v-container>
</template>
<script>
  import axios from 'axios'
export default {
    name: 'HelloWorld',
    data: () => ({
      loanAmount: '',
      predictedClass : ''
    }),
    methods: {
    submit () {
      axios.post('http://127.0.0.1:5000/predict', {
        LOAN_AMOUNT: this.loanAmount
      })
      .then((response) => {
        this.predictedClass = response.data.class
      })
    },
    clear () {
      this.loanAmount = ''
    }
  }
}
</script>